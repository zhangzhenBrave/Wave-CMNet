__all__ = ['WaveCMNet']


from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers.RevIN import RevIN
from typing import Optional
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from layers.Embed import DataEmbedding_inverted



class Conv_Block(nn.Module):
    def __init__(self, dim, kernel_size,d_model,dropout):
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=kernel_size,groups=dim,padding='same')
        # self.norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:# x: [Batch, Channel, Input length] [128, 30, 256]

        shortcut = x
        x = self.dwconv(x)
        # x = self.norm(x)
        x = shortcut + self.dropout(x)
        return x


class Conv_Encoder(nn.Module):
    def __init__(self,  conv_layers, norm_layer=None):
        super(Conv_Encoder, self).__init__()
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm = norm_layer
    def forward(self, x):
        # x [B, L, D]
        for conv_layer in self.conv_layers:
                x = conv_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.dropout = configs.dropout
        self.dim_outcomes=configs.dim_outcomes
        self.head_dropout=configs.head_dropout


        ##wavelet transform
        decompose_layer = configs.wavelet_j
        wave = configs.wavelet
        mode = 'symmetric'
        self.dwt = DWT1DForward(wave=wave, J=decompose_layer, mode=mode)
        self.idwt = DWT1DInverse(wave=wave)

        tmp1 = torch.randn(1, 1, self.seq_len)
        tmp1_yl, tmp1_yh = self.dwt(tmp1)
        tmp1_coefs = [tmp1_yl] + tmp1_yh

        tmp2 = torch.randn(1, 1, self.seq_len + self.pred_len)
        tmp2_yl, tmp2_yh = self.dwt(tmp2)
        tmp2_coefs = [tmp2_yl] + tmp2_yh
        assert decompose_layer + 1 == len(tmp1_coefs) == len(tmp2_coefs)

        ##revin层
        self.revin = configs.revin
        self.c_in = configs.c_in
        affine = configs.affine
        subtract_last = configs.subtract_last
        if self.revin: self.revin_layer = RevIN(self.dim_outcomes, affine=affine, subtract_last=subtract_last)

        ###学习变量之间attention
        self.variable_encodernets=Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.time_encodernets = Conv_Encoder(
            [
                Conv_Block(configs.c_in, configs.kernel_size,configs.d_model,configs.dropout)
                 for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        ###学习时序
        self.enc_embeddings= nn.ModuleList()
        # self.time_encodernets = nn.ModuleList()
        self.decodernets = nn.ModuleList()
        for i in range(decompose_layer + 1):
            self.enc_embeddings.append(DataEmbedding_inverted(tmp1_coefs[i].shape[-1], configs.d_model,configs.dropout))
            # self.time_encodernets.append(nn.Sequential( nn.Conv1d(in_channels=configs.c_in,out_channels=configs.c_in,kernel_size=configs.kernel_size,groups=configs.c_in,padding='same'),nn.LayerNorm(configs.d_model)))
            self.decodernets.append(nn.Sequential(torch.nn.LayerNorm(configs.d_model)
                ,nn.Linear(configs.d_model, (tmp2_coefs[i].shape[-1]) - (tmp1_coefs[i].shape[-1]), bias=True)))

        self.linear_t = nn.Linear(self.pred_len, configs.d_model)


    def model(self, coefs,t_pre):
        t_pre=self.linear_t(t_pre.permute(0,2,1))

        new_coefs = []
        # for coef,enc_embedding, encodernet,decodernet  in zip(coefs, self.enc_embeddings,self.encodernets,self.decodernets):
        # for coef, enc_embedding,time_encodernet,decodernet in zip(coefs, self.enc_embeddings,self.time_encodernets,self.decodernets):
        for coef, enc_embedding,decodernet in zip(coefs, self.enc_embeddings,self.decodernets):
            #词嵌入
            new_coef = enc_embedding(coef)
            ##编码器
            new_coef = self.time_encodernets(new_coef)
            new_coef, _ = self.variable_encodernets(new_coef)

            #new_coef = self.time_encodernets(new_coef)
            new_coef = torch.concat((t_pre, new_coef[:, -self.dim_outcomes:, :]), dim=1)

            ###解码器
            new_coef = decodernet(new_coef)
            new_coefs.append(new_coef[:, -self.dim_outcomes:, :])

        return new_coefs

    def forward(self, x_enc,t_pre):# x: [Batch, Input length, Channel][128, 336, 30]
      
        in_dwt = x_enc
        if self.revin:
            in_dwt  = self.revin_layer(in_dwt , 'norm')

        #小波域变换
        in_dwt = in_dwt.permute(0, 2, 1)                                                   #输出：in_dwt : [Batch,  Channel,Input length]
        yl, yhs = self.dwt(in_dwt)
        coefs = [yl] + yhs                                                                  #输出：coefs : [Batch,  Channel,Input length/2]
        ##############
        coefs_new = self.model(coefs,t_pre)

        coefs_idwt = []
        for i in range(len(coefs_new)):

             coefs_idwt.append(torch.cat((coefs[i][:, -self.dim_outcomes:, :], coefs_new[i]), 2))

        out = self.idwt((coefs_idwt[0], coefs_idwt[1:]))

        pred_out = out.permute(0, 2, 1)
        if self.revin:
            pred_out = self.revin_layer(pred_out, 'denorm')
        return pred_out

