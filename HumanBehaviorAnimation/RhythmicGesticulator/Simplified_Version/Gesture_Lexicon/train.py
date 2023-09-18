# region Import.

import os
import sys
import time
import torch
import json5

import torch.backends.cudnn as cudnn

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_sampler import *
from model import *

# endregion


class Trainer:
    def __init__(self, config, log_tag) -> None:
        # region Config.

        self.config = config

        # endregion

        # region Load data preprocessing config.

        with open(os.path.join(self.config["dir_data"], 'config.json5'), 'r') as f:
            self.config_data_preprocessing = json5.load(f)
        self.fps = self.config_data_preprocessing['fps']

        # endregion

        # region Log.

        date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.log_dir = os.path.join(self.config["dir_log"],
                                    log_tag + '_' + self.config['network']['name'] + '_' + date_time)
        os.makedirs(self.log_dir, exist_ok=True)
        
        tensorboard_log_dir = os.path.join(self.config["dir_log"],
                                           'Tensorboard_Log', log_tag + '_' + self.config['network']['name'] + '_' + date_time)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(logdir=tensorboard_log_dir)

        os.makedirs(os.path.join(self.log_dir, 'Checkpoints'), exist_ok=True)

        # endregion

        # region Copy config file to log dir.

        with open(os.path.join(self.log_dir, 'config.json5'), 'w') as f:
            json5.dump(self.config, f, indent=4)

        # endregion

        # region Device.

        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')

        # endregion
        
        # region Data loader.
        
        self.dataset_train = TrainingDataset(os.path.join(self.config["dir_data"], "train.npz"))
        self.data_loader_train = DataLoader(self.dataset_train, batch_size=self.config['batch_size'], shuffle=True)

        self.dataset_val = TrainingDataset(os.path.join(self.config["dir_data"], "valid.npz"))
        self.data_loader_val = DataLoader(self.dataset_val, batch_size=len(self.dataset_val), shuffle=False)
        
        # endregion

        cudnn.benchmark = True
        
        # region Network.

        if self.config['network']['name'] == 'Conv1d':
            self.net = Conv1d(self.config['network']['encoder_config'],
                              self.config['network']['decoder_config'])
        elif self.config['network']['name'] == 'Transformer':
            self.net = Transformer(**self.config['network']['hparams'])
        else:
            raise NotImplementedError

        self.net.to(self.device)
        
        # endregion
        
        # region Optimizer.

        if self.config['optimizer']['name'] == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                              lr=self.config['optimizer']['lr'],
                                              betas=self.config['optimizer']['betas'],
                                              eps=self.config['optimizer']['eps'],
                                              weight_decay=self.config['optimizer']['weight_decay'])
        elif self.config['optimizer']['name'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(params=self.net.parameters(),
                                               lr=self.config['optimizer']['lr'],
                                               betas=self.config['optimizer']['betas'],
                                               eps=self.config['optimizer']['eps'],
                                               weight_decay=self.config['optimizer']['weight_decay'])
        else:
            raise NotImplementedError

        # endregion
        
        # region Criterion.
        
        self.criterion = torch.nn.L1Loss()
        
        # endregion
        
    def train(self) -> None:
        # region Start timing.

        since = time.time()

        # endregion
        
        # region Epoch loop.
        
        num_epoch = self.config["num_epoch"]
        for epoch in range(1, num_epoch+1):
            print(f'\nEpoch: {epoch}/{num_epoch}')

            self.net.train()
            
            loss_train, loss_rot_train, loss_vel_train, loss_acc_train = 0., 0., 0., 0.
            counter = 0
            
            # region Data loader loop.
            
            pbar = tqdm(total=len(self.dataset_train), ascii=True)
            for _, batch in enumerate(self.data_loader_train):
                # region Prepare data.
                
                motion_block = batch["motion"].to(self.device)  # batch_size X dim_feat X time.
                
                # endregion
                
                # region Forward.
                
                self.optimizer.zero_grad()
                
                if self.config['network']['name'] in ['Conv1d', 'Transformer']:
                    _, motion_block_hat = self.net(motion_block)
                else:
                    raise NotImplementedError
                
                # endregion
                
                # region Loss and net weights update.
                
                loss_rot = self.criterion(motion_block, motion_block_hat)
                loss_vel = self.criterion(motion_block[:, :, 1:] - motion_block[:, :, :-1],
                                          motion_block_hat[:, :, 1:] - motion_block_hat[:, :, :-1])
                loss_acc = self.criterion(
                    motion_block[:, :, 2:] + motion_block[:, :, :-2] - 2 * motion_block[:, :, 1:-1],
                    motion_block_hat[:, :, 2:] + motion_block_hat[:, :, :-2] - 2 * motion_block_hat[:, :, 1:-1])
                loss = self.config["loss"]["rot"] * loss_rot + self.config["loss"]["vel"] * loss_vel + \
                       self.config["loss"]["acc"] * loss_acc
                
                loss.backward()
                self.optimizer.step()

                loss_rot_train += loss_rot.item() * motion_block.shape[0] * self.config["loss"]["rot"]
                loss_vel_train += loss_vel.item() * motion_block.shape[0] * self.config["loss"]["vel"]
                loss_acc_train += loss_acc.item() * motion_block.shape[0] * self.config["loss"]["acc"]
                loss_train += loss.item() * motion_block.shape[0]
                counter += motion_block.shape[0]
                
                # endregion
                
                # region Pbar update.

                pbar.set_description('lr: %s' % (str(self.optimizer.param_groups[0]['lr'])))
                pbar.update(motion_block.shape[0])
                
                # endregion
            
            pbar.close()
            
            # endregion
            
            # region Epoch loss and log.
            
            loss_train /= counter
            loss_rot_train /= counter
            loss_vel_train /= counter
            loss_acc_train /= counter
            
            print('Training',
                  f'Loss: {loss_train:.5f}',
                  f'Rot Loss: {loss_rot_train:.4f} /',
                  f'Vel Loss: {loss_vel_train:.4f} /',
                  f'Acc Loss: {loss_acc_train:.4f} /',
                  )

            self.writer.add_scalar(tag="Train/Loss", scalar_value=loss_train, global_step=epoch)
            self.writer.add_scalar("Rot_Loss/Train", loss_rot_train, epoch)
            self.writer.add_scalar("Vel_Loss/Train", loss_vel_train, epoch)
            self.writer.add_scalar("Acc_Loss/Train", loss_acc_train, epoch)
            
            # endregion
            
            # region Checkpoints.

            if epoch % self.config['checkpoint_save_epoch_num'] == 0:
                torch.save(self.net.state_dict(),
                           os.path.join(self.log_dir, 'Checkpoints', f'checkpoint_{epoch // 1000}k{epoch % 1000}.pth'))
            
            # endregion

            # region Validation.
            
            valid_num_epoch = self.config['valid_num_epoch']
            if epoch % valid_num_epoch == 0:
                self.net.eval()
                
                loss_valid, loss_rot_valid, loss_vel_valid, loss_acc_valid = 0., 0., 0., 0.
                counter = 0
                
                with torch.no_grad():
                    for _, batch in enumerate(self.data_loader_val):
                        motion_block = batch["motion"].to(self.device)  # batch_size X dim_feat X time.
                        
                        if self.config['network']['name'] in ['Conv1d', 'Transformer']:
                            _, motion_block_hat = self.net(motion_block)
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

                    self.writer.add_scalar(tag="Valid/Loss", scalar_value=loss_valid, global_step=epoch)
                    self.writer.add_scalar("Rot_Loss/Valid", loss_rot_valid, epoch)
                    self.writer.add_scalar("Vel_Loss/Valid", loss_vel_valid, epoch)
                    self.writer.add_scalar("Acc_Loss/Valid", loss_acc_valid, epoch)
            
            # endregion
            
        # endregion

        # region Save network.

        torch.save(self.net.state_dict(), os.path.join(self.log_dir, 'Checkpoints', 'trained_model.pth'))

        # endregion
        
        # region End timing.

        time_elapsed = time.time() - since
        print('\nTraining completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # endregion

        self.writer.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Must give config file path argument!')

    with open(sys.argv[1], 'r') as f:
        config = json5.load(f)

    log_tag = sys.argv[-1] if len(sys.argv) == 3 else ''

    trainer = Trainer(config, log_tag)
    trainer.train()