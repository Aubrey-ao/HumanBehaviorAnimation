# region Import.

import os
import sys
import torch
import json5

import numpy as np

module_path = os.path.dirname(os.path.abspath(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)

from data_sampler import *
from model import *
from torch.utils.data import DataLoader

# endregion


class Inference:
    def __init__(self, 
                 path_data : str, 
                 path_pretrained_net : str,
                 device : str, 
                 path_config_train : str) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        with open(path_config_train, "r") as f:
            self.config = json5.load(f)

        self.dataset = TrainingDataset(path_data)
        self.data_loader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        if self.config['network']['name'] == 'Conv1d':
            self.net = Conv1d(self.config['network']['encoder_config'],
                              self.config['network']['decoder_config'])
        elif self.config['network']['name'] == 'Transformer':
            self.net = Transformer(**self.config['network']['hparams'])
        else:
            raise NotImplementedError

        self.net.load_state_dict(torch.load(path_pretrained_net, map_location=torch.device("cpu")))
        self.net.to(self.device)
        
        self.criterion = torch.nn.L1Loss()
        
    
    def infer(self) -> np.ndarray:
        self.net.eval()
                
        loss_valid, loss_rot_valid, loss_vel_valid, loss_acc_valid = 0., 0., 0., 0.
        counter = 0
        with torch.no_grad():
            latent_codes =[]
            for batch in self.data_loader:
                motion_block = batch['motion'].to(self.device)  # batch_size X dim_feat X time.
            
                if self.config['network']['name'] in ['Conv1d', 'Transformer']:
                    latent_code, motion_block_hat = self.net(motion_block)
                else:
                    raise NotImplementedError

                loss_rot = self.criterion(motion_block, motion_block_hat)
                loss_vel = self.criterion(motion_block[:, :, 1:] - motion_block[:, :, :-1],
                                          motion_block_hat[:, :, 1:] - motion_block_hat[:, :, :-1])
                loss_acc = self.criterion(
                    motion_block[:, :, 2:] + motion_block[:, :, :-2] - 2 * motion_block[:, :, 1:-1],
                    motion_block_hat[:, :, 2:] + motion_block_hat[:, :, :-2] - 2 * motion_block_hat[:, :, 1:-1])
                loss = self.config["loss"]["rot"] * loss_rot + self.config["loss"]["vel"] * loss_vel + \
                       self.config["loss"]["acc"] * loss_acc

                loss_rot_valid += loss_rot.item() * motion_block.shape[0] * self.config["loss"]["rot"]
                loss_vel_valid += loss_vel.item() * motion_block.shape[0] * self.config["loss"]["vel"]
                loss_acc_valid += loss_acc.item() * motion_block.shape[0] * self.config["loss"]["acc"]
                loss_valid += loss.item() * motion_block.shape[0]
                counter += motion_block.shape[0]

                latent_code = latent_code.detach().cpu().numpy()
                latent_code = np.concatenate(np.split(latent_code, self.dataset.index.shape[1], axis=0), axis=2)
                latent_code = np.transpose(latent_code, (0, 2, 1))  # num_clips X num_blocks X dim_feat.
                latent_codes.append(latent_code)

        loss_valid /= counter
        loss_rot_valid /= counter
        loss_vel_valid /= counter
        loss_acc_valid /= counter
        print('Validation',
              f'Loss: {loss_valid:.5f}',
              f'Rot Loss: {loss_rot_valid:.4f} /',
              f'Vel Loss: {loss_vel_valid:.4f} /',
              f'Acc Loss: {loss_acc_valid:.4f} /',
              )

        latent_codes = np.concatenate(latent_codes, axis=0)  # num_clips X num_blocks X dim_feat.
        
        return latent_codes


if __name__ == "__main__":
    path_data = ""
    path_pretrained_net = ""
    path_config_train = ""

    device = "cuda:0"
    
    inference = Inference(path_data, path_pretrained_net, device, path_config_train)
    
    latent_code = inference.infer()
    
    print(latent_code.shape)