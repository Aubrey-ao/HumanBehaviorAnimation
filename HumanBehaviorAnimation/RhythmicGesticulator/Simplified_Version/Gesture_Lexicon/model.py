# region Import.

import torch

import numpy as np
import torch.nn as nn

from typing import List

# endregion


__all__ = ["Conv1d", "Transformer"]


class Conv1d(nn.Module):
    def __init__(self, 
                 encoder_config : List[List[int]], 
                 decoder_config : List[List[int]]) -> None:
        super().__init__()
        
        num_layers = len(encoder_config)
        
        modules = []
        for i, c in enumerate(encoder_config):
            modules.append(nn.Conv1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))
            
            if i < (num_layers-1):
                modules.append(nn.BatchNorm1d(c[1]))
                modules.append(nn.LeakyReLU(1.0, inplace=True))
        
        self.encoder = nn.Sequential(*modules)
        
        
        num_layers = len(decoder_config)
        
        modules = []
        for i, c in enumerate(decoder_config):
            modules.append(nn.ConvTranspose1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))
            
            if i < (num_layers-1):
                modules.append(nn.BatchNorm1d(c[1]))
                modules.append(nn.LeakyReLU(1.0, inplace=True))
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, x : torch.Tensor):
        """
        x : (batch_size, dim_feat, time).
        """
        
        latent_code = self.encoder(x)
        
        return latent_code, self.decoder(latent_code)


# --------------------------------------------------------
# Reference: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# --------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100) -> None:
        super().__init__()

        assert d_model % 2 == 0

        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.pe.data[0, :, 0::2].copy_(torch.sin(position * div_term))
        self.pe.data[0, :, 1::2].copy_(torch.cos(position * div_term))

    def forward(self, x) -> torch.Tensor:
        """
        x: [N, L, D]
        """
        x = x + self.pe[:, :x.shape[1], :]

        return self.dropout(x)


# --------------------------------------------------------
# Reference: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, mo_dim, lxm_dim,
                 embed_dim=512, depth=6, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
                 mlp_ratio=4, activation='gelu', norm_layer=nn.LayerNorm, dropout=0.1) -> None:
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.encoder_embed = nn.Linear(mo_dim, embed_dim, bias=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEncoding(embed_dim, dropout)  # hack: max len of position is 100

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim,
                                                                        nhead=num_heads,
                                                                        dim_feedforward=int(mlp_ratio*embed_dim),
                                                                        dropout=dropout,
                                                                        activation=activation,
                                                                        batch_first=True),
                                             num_layers=depth)

        self.norm = norm_layer(embed_dim)
        self.lxm_embed = nn.Linear(embed_dim, lxm_dim, bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(lxm_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = PositionalEncoding(decoder_embed_dim, dropout)  # hack: max len of position is 100

        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=decoder_embed_dim,
                                                                        nhead=decoder_num_heads,
                                                                        dim_feedforward=int(mlp_ratio*decoder_embed_dim),
                                                                        dropout=dropout,
                                                                        activation=activation,
                                                                        batch_first=True),
                                             num_layers=decoder_depth)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, mo_dim, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Parameter
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed.pe, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed.pe, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    def forward_encoder(self, x):
        """
        x: [N, D, L]
        """
        x = torch.einsum('NDL->NLD', x)

        # embed motion sequence
        x = self.encoder_embed(x)

        # append cls token
        cls_token = self.cls_token.repeat(x.shape[0], x.shape[1], 1)
        x = torch.cat([cls_token, x], dim=1)

        # add pos embed
        x = self.pos_embed(x)

        # apply Transformer blocks
        x = self.encoder(x)
        x = self.norm(x)
        x = self.lxm_embed(x)

        lxm = torch.einsum('NLD->NDL', x[:, :1, :])

        return lxm

    def forward_decoder(self, lxm, mo_len):
        '''
        lxm: [N, D, L]
        '''
        lxm = torch.einsum('NDL->NLD', lxm)

        # embed lexeme
        x = self.decoder_embed(lxm)

        # append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], mo_len, 1)
        x = torch.cat([x, mask_tokens], dim=1)

        # add pos embed
        x = self.decoder_pos_embed(x)

        # add Transformer blocks
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove lxm token
        x = x[:, 1:, :]

        x = torch.einsum('NLD->NDL', x)

        return x

    def forward(self, x):
        """
        x: [N, D, L]
        """
        _, _, L = x.shape

        lxm = self.forward_encoder(x)
        x_hat = self.forward_decoder(lxm, L)

        return lxm, x_hat

# region Test.

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # region Conv1d
    
    # encoder_config = [
    #     [42, 64, 5, 1, 0],
    #     [64, 128, 4, 2, 0],
    #     [128, 156, 4, 1, 0],
    #     [156, 192, 4, 1, 0]
    # ]
    # decoder_config = [
    #     [192, 156, 4, 1, 0],
    #     [156, 128, 4, 1, 0],
    #     [128, 64, 4, 2, 0],
    #     [64, 42, 5, 1, 0]
    # ]
    #
    # conv_1d = Conv1d(encoder_config, decoder_config).to(device)
    #
    # x = torch.randn((5, 42, 20)).to(device)
    # motif, x_hat = conv_1d(x)
    #
    # print(motif.shape, x_hat.shape)
    
    # endregion

    # region Transformer

    # model = Transformer(48, 96).to(device)
    #
    # x = torch.randn((5, 48, 10)).to(device)  # [N, D, L]
    #
    # lexeme, x_hat = model(x)
    #
    # print(lexeme.shape)
    # print(x_hat.shape)

    # endregion
    
    # region network statistics

    # def get_parameter_number(model):
    #     total_num = sum(p.numel() for p in model.parameters())
    #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
    #
    # print(get_parameter_number(model))

    # endregion

# endregion