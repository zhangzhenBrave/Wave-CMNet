
__all__ = ['DCDN']

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.transformers1new import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
import math
from torch.nn.utils import weight_norm


def get_initialiser(name='xavier'):
    if name == "orthogonal":
        return nn.init.orthogonal_
    elif name == "xavier":
        return nn.init.xavier_uniform_
    elif name == "kaiming":
        return nn.init.kaiming_uniform_
    elif name == "none":
        pass
    else:
        raise Exception("Unknown init method")


def get_activation(name: str) -> nn.Module:
    if name == "rrelu":
        return nn.RReLU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    else:
        raise Exception("Unknown activation")


def create_batch_norm_1d_layers(num_layers: int, dim_hidden: int):
    batch_norm_layers = nn.ModuleList()
    for i in range(num_layers - 1):
        batch_norm_layers.append(nn.BatchNorm1d(num_features=dim_hidden))
    return batch_norm_layers


def create_linear_layers(
        num_layers: int, dim_input: int, dim_hidden: int, dim_output: int
):
    linear_layers = nn.ModuleList()
    # Input layer
    linear_layers.append(nn.Linear(in_features=dim_input, out_features=dim_hidden))
    # Hidden layers
    for i in range(1, num_layers - 1):
        linear_layers.append(nn.Linear(in_features=dim_hidden, out_features=dim_hidden))
    # Output layer
    linear_layers.append(nn.Linear(dim_hidden, dim_output))
    return linear_layers


def init_layers(initialiser_name: str, layers: nn.ModuleList):
    initialiser = get_initialiser(initialiser_name)
    for layer in layers:
        initialiser(layer.weight)


class Embeddings(nn.Module):
    def __init__(self, emb_size, act=None, initrange=0.01, res=0, time_step=96):
        super(Embeddings, self).__init__()
        self.treat_weight = nn.Linear(time_step, emb_size)
        self.initrange = initrange
        self.res = res
        if res:
            self.emb_size = emb_size + 1
        else:
            self.emb_size = emb_size
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
            self.act = nn.Sigmoid()
        else:
            self.act = None
        self.init_weights()

    def forward(self, tokens):
        ebd = self.treat_weight(tokens.to(torch.float32))  # tokens:[batch_size, 288], ebd:[batch_size, 288, 288]

        print('ebd:', ebd.size())
        if self.res:
            ebd = torch.cat([torch.ones(ebd.shape[0], 1).cuda(), ebd], dim=-1)
        if self.act is None:
            return ebd
        return self.act(ebd)  # ebd:[batch_size, 288, 289]

    def init_weights(self) -> None:
        self.treat_weight.weight.data.normal_(0, self.initrange)
        self.treat_weight.bias.data.zero_()



class MLP(nn.Module):
    def __init__(
            self,
            dim_input: int,
            dim_hidden: int,
            dim_output: int,
            num_layers=1,
            batch_norm=True,
            initialiser='xavier',
            dropout=0.0,
            activation='relu',
            leaky_relu=0.1,
            is_output_activation=True,
    ):
        super().__init__()
        self.layers = create_linear_layers(
            num_layers=num_layers,
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
        )
        init_layers(initialiser_name=initialiser, layers=self.layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.batch_norm_layers = (
            create_batch_norm_1d_layers(num_layers=num_layers, dim_hidden=192)
            if batch_norm
            else None
        )
        self.activation_function = get_activation(
            name=activation
        )
        self.is_output_activation = is_output_activation

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            # print(i)
            # print(x)
            x = self.activation_function(x)
            if self.batch_norm_layers:
                x = self.batch_norm_layers[i](x)
            if self.dropout:
                x = self.dropout(x)
        x = self.layers[-1](x)
        if self.is_output_activation:
            x = self.activation_function(x)
        return x


class LSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, num_layers, timesteps=96):
        super(LSTM, self).__init__()

        self.seqLen = timesteps
        self.num_layers = num_layers
        self.hiddenSize = hiddenSize
        self.lstm = nn.LSTM(input_size=inputSize, hidden_size=hiddenSize, num_layers=num_layers, batch_first=False,
                            bidirectional=False)

    def forward(self, x):  # x.shape = [bsz, timesteps, inputSize]
        bsz = x.shape[1]

        #x = x.permute(1, 0, 2)
        h0 = torch.rand(self.num_layers, bsz, self.hiddenSize).to('cuda')
        c0 = torch.rand(self.num_layers, bsz, self.hiddenSize).to('cuda')
        # print('x:', x.size())
        # print('h0:', h0.size())

        out, (out_h, out_c) = self.lstm(x, (h0, c0))

        return out, out_h, out_c

class  Model(nn.Module):
    def __init__(self,configs):
        super( Model, self).__init__()

        self.inp_dim=configs.dim_vitals+configs.dim_outcomes-1
        self.time_steps = configs.seq_len
        num_layers = 3
        self.embed_size = configs.pred_len
        self.pred_len= configs.pred_len
        self.dim_outcomes=configs.dim_outcomes
        num_t = 4
        num_cov = self.embed_size
        dropout = 0.1
        att_layers_en = 4
        att_layers_de = 3
        activation = 'relu'
        n_heads=2

        self.use_lstm = True
        self.rnn_hiddenSize = 96

        self.project_inp = nn.Linear(self.time_steps, self.embed_size)

        encoder_layers = TransformerEncoderLayer(d_model=self.embed_size, nhead=n_heads, dim_feedforward=configs.d_ff,
                                                 dropout=dropout, num_cov=num_cov, activation=activation)
        self.encoder = TransformerEncoder(encoder_layers, att_layers_en)  # att_layers 编码器中的子编码器层数。


        decoder_layers = TransformerDecoderLayer(self.embed_size, nhead=n_heads, dim_feedforward=configs.d_ff, dropout=dropout,
                                                 num_t=num_t, activation=activation)
        self.decoder = TransformerDecoder(decoder_layers, att_layers_de)


        if self.use_lstm:
            self.lstm = LSTM(inputSize=64, hiddenSize=90, num_layers=num_layers)

        self.Q_0 = MLP(
            dim_input=90,
            dim_hidden=45,
            dim_output=configs.dim_outcomes,
            is_output_activation=True,
        )

    def forward(self, x_enc,t_pre):#x [128, 336, 1+24 +5] t_pre ,[128, 96, 1]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x=x_enc[:,:,1:]
        inp = x.permute(2, 0, 1)  # x[bsz, seq_len, inp_dim=24] --> inp:[inp_dim, bsz, seq_len]
        inp = self.project_inp(inp)  # inp:[inp_dim, bsz, seq_len] --> inp:[inp_dim, bsz, embed_size]
        # inp.shape = torch.Size([29, 128, 256])
        memory = self.encoder(inp)  # memory: [inp_dim, bsz, embed_size]--> memory: [inp_dim, batch_size, embedsize]
        # memory.shape = torch.Size([29, 128, 256])

        # fai = memory.permute(1, 2, 0)  # [batch_size, embed_size, inp_dim]

        out = self.decoder(t_pre.permute(2, 0, 1), memory)  # out:[t_dim, bz, embedsize] (29, 64, 96)
        #print('out',out.shape)out torch.Size([64, 128, 96])
        if self.use_lstm:
            out, out_h, out_c = self.lstm(out.permute(2, 1, 0))  # (77, bz, 96)

        # print('out：', out.size())

        Q_0 = self.Q_0(out)
        dec_out = Q_0.permute(1, 0, 2)
        dec_out = dec_out * \
                  (stdev[:, 0, -self.dim_outcomes:].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, -self.dim_outcomes:].unsqueeze(1).repeat(
                      1, self.pred_len, 1))

        return  dec_out

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():

            if isinstance(m, nn.Linear):
                if m.in_features == 1:
                    continue
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
