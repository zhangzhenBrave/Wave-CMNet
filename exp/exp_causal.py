
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import WaveCMNet, PatchTST,MICN,FreTS,DCDN,TimesNet,FEDformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from thop import profile
warnings.filterwarnings('ignore')

class moving_avg(nn.Module):
    def __init__(self):
        super(moving_avg, self).__init__()
    def forward(self, x, kernel_size):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            convert_numpy = True
            x = torch.tensor(x)
        else:
            convert_numpy = False
        x = nn.functional.avg_pool1d(x.permute(0, 2, 1), kernel_size, kernel_size)
        x = x.permute(0, 2, 1)
        if convert_numpy:
            x = x.numpy()
        return x
class Exp_causal(Exp_Basic):
    def __init__(self, args):
        super(Exp_causal, self).__init__(args)
        self.mv = moving_avg()
    def _build_model(self):
        model_dict = {
            'WaveCMNet':WaveCMNet,
            'PatchTST': PatchTST,
            'TimesNet':TimesNet,
            'FEDformer':FEDformer,
            'MICN': MICN,
            'FreTSLinear':FreTS,
            'MICN': MICN,
            'DCDN':DCDN
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        #criterion =nn.HuberLoss(delta=1.0)
        return criterion

    def _get_profile(self,name, model):
        if name in  ['PatchTST',  'FreTSLinear','TimesNet']:
            _input=torch.randn(self.args.batch_size, self.args.seq_len, self.args.c_in).to(self.device)
            macs, params = profile(model, inputs=(_input,))
        elif name in ['WaveCMNet','DCDN' ]:
            _input=torch.randn(self.args.batch_size, self.args.seq_len, self.args.c_in).to(self.device)
            _treat=torch.randn(self.args.batch_size, self.args.pred_len, self.args.dim_treatments).to(self.device)
            macs, params = profile(model, inputs=(_input, _treat))
        else:
            _input=torch.randn(self.args.batch_size, self.args.seq_len, self.args.c_in).to(self.device)
            dec_inp = torch.randn(self.args.batch_size, self.args.pred_len, self.args.c_in).to(self.device)
            dec_inp = torch.cat([ dec_inp, dec_inp], dim=1).float().to(self.device)
            macs, params = profile(model, inputs=(_input,  dec_inp))
        print('FLOPs: ', macs)
        print('params: ', params)
        return macs, params

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_t) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_t = batch_t.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if  self.args.model  in ['PatchTST', 'FreTSLinear','TimesNet']:
                             
                             outputs = self.model(batch_x)
                        elif self.args.model in ['WaveCMNet','DCDN' ]:
        
                            outputs = self.model(batch_x, batch_t)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x,  dec_inp)[0]
                            else:
                                outputs = self.model(batch_x, dec_inp)


                else:
                    if  self.args.model  in ['PatchTST',  'FreTSLinear','TimesNet']:
                        outputs = self.model(batch_x)
                    elif self.args.model in ['WaveCMNet','DCDN']:
        
                        outputs = self.model(batch_x, batch_t)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x,  dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)

                if self.args.features == 'MLS':  # MLS:multivariate predict last multivariate
                    f_dim = -self.args.dim_outcomes
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self._get_profile(self.args.model,self.model)
        print('Trainable parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        train_losses = []
        val_losses = []
        test_losses = []
        speeds=[]
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_t) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_t = batch_t.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if  self.args.model  in ['PatchTST', 'FreTSLinear','TimesNet']:
                            outputs = self.model(batch_x)
                        elif self.args.model in ['WaveCMNet','DCDN']:
        
                            outputs = self.model(batch_x,batch_t)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x,  dec_inp)[0]
                            else:
                                outputs = self.model(batch_x, dec_inp)

                        if self.args.features == 'MLS':  # MLS:multivariate predict last multivariate
                            f_dim = -self.args.dim_outcomes

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # print(outputs.shape,batch_xy.shape)
                        # loss = criterion(outputs, batch_xy)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                else:
                        if  self.args.model  in ['PatchTST', 'FreTSLinear','TimesNet']:
                             outputs = self.model(batch_x)
                        elif self.args.model in ['WaveCMNet','DCDN']:
        
                            outputs = self.model(batch_x, batch_t)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, dec_inp)[0]

                            else:
                                outputs = self.model(batch_x, dec_inp)

                        if self.args.features == 'MLS':  # MLS:multivariate predict last multivariate
                            f_dim = -self.args.dim_outcomes

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # print(outputs.shape,batch_xy.shape)
                        # loss = criterion(outputs, batch_xy)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            speeds.append(time.time() - epoch_time)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(vali_loss)
            test_losses.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        folder_path = './test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'train_losses.npy', train_losses)
        np.save(folder_path + 'val_losses.npy', val_losses)
        np.save(folder_path + 'test_losses.npy', test_losses)

        plt.plot(train_losses, 'b', label='Training Loss')
        plt.plot( val_losses, 'r', label='Validation Loss')
        plt.plot( test_losses, 'g', label='Test Loss')
        plt.title('Training, Validation, and Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_curve.pdf', format='pdf')
        print('speed',np.average(speeds))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds =  []
        trues = []
        inputx =  []
        inputt =[]
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,batch_t ) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_t = batch_t.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if  self.args.model  in ['PatchTST', 'FreTSLinear','TimesNet']:
                              outputs = self.model(batch_x)
                        elif self.args.model in ['WaveCMNet','DCDN']:
        
                            outputs = self.model(batch_x, batch_t)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, dec_inp)[0]
                            else:
                                outputs = self.model(batch_x, dec_inp)
                else:
                    if  self.args.model  in ['PatchTST',  'FreTSLinear','TimesNet']:
                        outputs = self.model(batch_x)
                    elif self.args.model in ['WaveCMNet','DCDN' ]:
        
                        outputs = self.model(batch_x, batch_t)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)


                if self.args.features == 'MLS':  # MLS:multivariate predict last multivariate
                    f_dim = -self.args.dim_outcomes
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues .append( true)
                inputx .append( batch_x.detach().cpu().numpy())
                inputt.append( batch_t.detach().cpu().numpy())

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        inputt= np.concatenate(inputt, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        np.save(folder_path + 't.npy', inputt)
        return

   