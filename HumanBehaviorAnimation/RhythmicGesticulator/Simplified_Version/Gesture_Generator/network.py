# region Import.

import torch

import torch.nn as nn

from torch.nn import functional as F

# endregion


__all__ = ['MotionGenerator_RNN']


class MotionGenerator_RNN(nn.Module):
    def __init__(self,
                 aud_dim, aud_hid_dim, aud_embed_dim,
                 mo_dim,
                 lxm_dim,
                 rnn_hid_dim, rnn_out_dim, rnn_depth):
        super().__init__()

        self.audio_encoder = AudioEncoder_Conv1d(aud_dim, aud_hid_dim, aud_embed_dim)

        self.motion_decoder = GRUDecoder(mo_dim, aud_embed_dim, lxm_dim, rnn_hid_dim, rnn_out_dim, rnn_depth)
        self.cell_state_init = CellStateInitializer(mo_dim+lxm_dim, rnn_hid_dim, rnn_depth)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    def forward(self, aud, mo, lxm):
        """
        aud, mo: [N, D, L]
        lxm: [N, D, B]. B: num_block
        """
        N, D, L = aud.shape
        B = lxm.shape[-1]
        BL = L // B  # BL: block length

        # initialize the hidden state of rnn
        cell_state = self.cell_state_init(mo[:, :, BL-1], lxm[:, :, 0])

        mo_hat = []  # without the first and the last blocks. element shape: [N, D, 1]
        for b_idx in range(B-2):
            aud_embed_b = self.audio_encoder(aud[:, :, (b_idx+0)*BL: (b_idx+3)*BL])

            for f in range(BL):  # f: frame
                # get previous pose
                pre_mo = mo[:, :, BL-1] if b_idx == 0 and f == 0 else mo_hat[-1][:, :, 0]

                # get current audio embedding and lexeme
                cur_aud_embed = aud_embed_b[:, :, f]
                cur_lxm = lxm[:, :, b_idx+1]

                mo_pred, cell_state = self.motion_decoder(pre_mo, cur_aud_embed, cur_lxm, cell_state)

                mo_hat.append(mo_pred.unsqueeze(-1))

        mo_hat = torch.cat(mo_hat, dim=-1)

        return mo_hat


class AudioEncoder_Conv1d(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super().__init__()

        self.layer0 = nn.Conv1d(in_dim, hidden_dim, 7, 1, 0)  # L: 30 -> 24
        self.layer1 = nn.Conv1d(hidden_dim, hidden_dim, 5, 1, 0)  # L: 24 -> 20
        self.layer2 = nn.Conv1d(hidden_dim, out_dim, 4, 2, 1)  # L: 20 -> 10
        self.layer3 = nn.Linear(out_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [N, D, L]
        """

        x = F.gelu(self.layer0(x))
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))

        x = torch.einsum('NDL->NLD', x)

        x = F.gelu(self.layer3(x))
        x = self.norm(x)

        x = torch.einsum('NLD->NDL', x)

        return x


class GRUDecoder(nn.Module):
    def __init__(self, mo_dim, aud_embed_dim, lxm_dim, hidden_dim, out_dim, depth):
        super().__init__()

        in_dim = mo_dim + aud_embed_dim + lxm_dim

        self.embed_in = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(in_dim+hidden_dim, hidden_dim, depth, batch_first=True)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.embed_pred = nn.Linear(hidden_dim, out_dim)

    def forward(self, mo, aud_embed, lxm, cell_state):
        """
        mo, aud_embed, lxm: [N, D]
        cell_state: [P, N, D]
        """
        in_embed = F.gelu(self.embed_in(torch.cat([mo, aud_embed, lxm], dim=-1)))
        cell_out, cell_state = self.gru(
            torch.cat([in_embed, mo, aud_embed, lxm], dim=-1).unsqueeze(1), cell_state  # add an input embedding (redundancy) into the gru input
        )
        out = self.decoder_norm(cell_out)
        out = self.embed_pred(out.squeeze(1))

        return out, cell_state


class CellStateInitializer(nn.Module):
    def __init__(self, in_dim, hidden_dim, rnn_depth):
        super().__init__()

        self.rnn_depth = rnn_depth

        self.layer0 = nn.Linear(in_dim, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim*rnn_depth)

    def forward(self, mo, lxm):
        """
        mo, lxm: [N, D]
        """
        x = F.gelu(self.layer0(torch.cat([mo, lxm], dim=-1)))
        x = F.gelu(self.layer1(x))
        x = self.layer2(x)

        x = x.reshape(x.shape[0], self.rnn_depth, -1)
        x = torch.einsum('NPD->PND', x).contiguous()  # P: depth

        return x


# region Test.

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # region audio encoder (conv1d)

    # model = AudioEncoder_Conv1d(43, 64, 64).to(device)
    #
    # x = torch.rand([5, 43, 30]).to(device)  # [N, D, L]
    # y = model(x)
    #
    # print(y.shape)

    # endregion

    # region rnn cell state initializer

    # model = CellStateInitializer(48+96, 1024, 2).to(device)
    #
    # x = torch.rand([5, 48]).to(device)  # [N, D]
    # x1 = torch.rand([5, 96]).to(device)  # [N, D]
    # y = model(x, x1)
    #
    # print(y.shape)

    # endregion

    # region gru decoder

    # model = GRUDecoder(48, 64, 96, 1024, 48, 2).to(device)
    #
    # x = torch.rand([5, 48]).to(device)  # [N, D]
    # x1 = torch.rand([5, 64]).to(device)  # [N, D]
    # x2 = torch.rand([5, 96]).to(device)  # [N, D]
    # x3 = torch.rand([2, 5, 1024]).to(device)  # [N, D]
    # y, y1 = model(x, x1, x2, x3)
    #
    # print(y.shape)
    # print(y1.shape)

    # endregion

    # region motion decoder rnn

    # model = MotionGenerator_RNN(43, 64, 64,
    #                             48,
    #                             96,
    #                             1024, 48, 2).to(device)
    #
    # x = torch.rand([5, 43, 100]).to(device)  # [N, D, L]
    # x1 = torch.rand([5, 48, 100]).to(device)  # [N, D, L]
    # x2 = torch.rand([5, 96, 10]).to(device)  # [N, D, L]
    #
    # y = model(x, x1, x2)
    #
    # print(y.shape)

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
