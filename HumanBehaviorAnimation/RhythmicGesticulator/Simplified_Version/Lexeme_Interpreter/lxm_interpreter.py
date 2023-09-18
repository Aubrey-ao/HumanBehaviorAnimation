# region Import.

import torch

import numpy as np
import torch.nn as nn

from torch.nn import functional as F

# endregion


__all__ = ['LxmInterpreter']


class LxmInterpreter(nn.Module):
    def __init__(self, lxc_size, train_block_num, train_block_len,
                 aud_dim, embed_dim, num_heads, encoder_depth, decoder_depth,
                 mlp_ratio=4, activation='gelu', norm_layer=nn.LayerNorm, dropout=0.0):
        super().__init__()

        self.train_block_num = train_block_num
        self.train_block_len = train_block_len

        self.encoder_embed = nn.Linear(aud_dim, embed_dim, bias=True)
        self.encoder_pos_embed = PositionalEncoding(embed_dim, dropout)  # hack: max len of position is 200

        self.decoder_embed = nn.Embedding(lxc_size, embed_dim)
        self.decoder_pos_embed = PositionalEncoding(embed_dim, dropout)  # hack: max len of position is 200

        self.transformer = nn.Transformer(embed_dim, num_heads, encoder_depth, decoder_depth, int(mlp_ratio*embed_dim),
                                          dropout, activation, batch_first=True)

        self.decoder_norm = norm_layer(embed_dim)
        self.cls_head = nn.Linear(embed_dim, lxc_size, bias=False)

        # initialization
        # initialize nn.Parameter
        torch.nn.init.normal_(self.encoder_pos_embed.pe, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed.pe, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    def _build_mask(self, en_L, en_BL, de_L):
        # encoder mask
        en_mask = torch.triu(torch.ones(en_L, en_L))
        en_mask[
            torch.arange(en_L).unsqueeze(1).repeat(1, en_BL).reshape(-1),
            torch.arange(en_L).reshape(-1, en_BL).repeat_interleave(en_BL, 0).reshape(-1)
        ] = 0
        en_mask = en_mask.float().masked_fill(en_mask == 1., float('-inf')).masked_fill(en_mask == 0., float(0.))

        # memory mask
        mem_mask = torch.zeros(de_L, en_L).float()
        mem_mask.copy_(en_mask[torch.arange(0, en_L, en_BL), :])

        # decoder mask
        de_mask = nn.Transformer.generate_square_subsequent_mask(de_L)

        return en_mask, de_mask, mem_mask

    def forward(self, aud, lxm_idx):
        '''
        aud: [N, L, D]
        lxm_idx: [N, B]. B: num_block
        '''
        device = aud.device
        N, L, D = aud.shape
        B = lxm_idx.shape[-1]
        BL = L // B  # block len

        aud_embed = self.encoder_embed(aud)
        aud_embed = self.encoder_pos_embed(aud_embed)

        lxm_embed = self.decoder_embed(lxm_idx)  # [N, B, D]
        lxm_embed = self.decoder_pos_embed(lxm_embed)

        masks = self._build_mask(L, BL, B)

        out = self.transformer(aud_embed, lxm_embed, *[m.to(device) for m in masks])

        out = self.decoder_norm(out)
        logits = self.cls_head(out)

        return logits

    def generate(self, aud, lxm_idx, temperature=1.0, do_sample=False, top_k=None):
        '''
        aud: [N, L, D]
        lxm_idx: [N, 1]
        '''
        BL = self.train_block_len
        B = aud.shape[1] // BL
        max_B = self.train_block_num

        logits_pred = []
        for B_idx in range(B):
            aud_cond = aud[:, :(B_idx+1)*BL, :] if B_idx < max_B else aud[:, (B_idx+1-max_B)*BL: (B_idx+1)*BL, :]
            lxm_idx_cond = lxm_idx if lxm_idx.shape[-1] <= max_B else lxm_idx[:, -max_B:]

            logits = self(aud_cond, lxm_idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                lxm_idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, lxm_idx_next = torch.topk(probs, k=1, dim=-1)

            logits_pred.append(logits.unsqueeze(1))
            lxm_idx = torch.cat([lxm_idx, lxm_idx_next], dim=1)
        logits_pred = torch.cat(logits_pred, dim=1)

        return lxm_idx[:, 1:], logits_pred


# --------------------------------------------------------
# Reference: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# --------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200) -> None:
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


# region Test

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = LxmInterpreter(2048, 10, 10, 80, 256, 4, 3, 3).to(device)

    x = torch.randn((5, 100, 80)).to(device)
    idx = torch.randint(2048, (5, 10)).to(device)

    logits = model(x, idx)
    print(logits.shape)

    idx_init = torch.randint(2048, (5, 1)).to(device)
    idx_pred, logits = model.generate(x, idx_init, top_k=3)
    print(idx_pred.shape, logits.shape)
    idx_pred, logits = model.generate(x, idx_init, do_sample=True)
    print(idx_pred.shape, logits.shape)

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    print(get_parameter_number(model))

# endregion