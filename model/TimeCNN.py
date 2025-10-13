import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class FFN(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(FFN, self).__init__()
        self.d_model = d_model  
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)
    
class CrossCNN(nn.Module):
    def __init__(self, seq_len, c_in, dropout=0.1):
        super(CrossCNN, self).__init__()

        #TimeCNN
        self.crosscnn = nn.Conv1d(seq_len, seq_len, c_in, groups=seq_len)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #input  x [B L N]
        #output x [B L N]

        
        #TimeCNN
        pad = x[:, :, 1:]
        pad = torch.cat([pad, x], dim=-1)
        x = self.dropout(self.crosscnn(pad)) + x

        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        seq_len = configs.seq_len
        self.pred_len = pred_len = configs.pred_len
        self.d_model = d_model = configs.d_model
        dropout = configs.dropout
        d_ff = configs.d_ff
        c_in = configs.enc_in
        self.use_norm = configs.use_norm

        #crossCNN
        self.crosscnn = CrossCNN(seq_len, c_in, configs.conv_dropout)

        #Embedding
        self.emb = nn.Linear(seq_len, d_model)

        #FFN
        self.layers = nn.ModuleList(FFN(d_model=d_model, d_ff=d_ff, dropout=dropout, activation="gelu") for _ in range(configs.e_layers))

        #prediction
        self.linear_pred = nn.Linear(d_model, pred_len)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x_enc
        B, L, N = x.shape

        # x: [B L N]
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        #refine cross-variable interaction on time point
        x = self.crosscnn(x)

        #embedding
        x = self.emb(x.permute(0,2,1))

        #FFN
        for layer in self.layers:
            x = layer(x)

        #prediction
        x = self.linear_pred(x).permute(0,2,1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return x # [Batch, Output length, Channel]

