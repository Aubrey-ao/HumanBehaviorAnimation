# region Import.

import os
import sys
import time
import torch
import json5

import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_importor import *
from lxm_interpreter import *
from tools import *

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
        self.uniform_len = self.config_data_preprocessing["uniform_len"]
        self.num_blocks = self.config_data_preprocessing["num_blocks_per_clip"]
        self.lxc_size = self.config_data_preprocessing["lexicon_size"]

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
        self.data_loader_val = DataLoader(self.dataset_val, batch_size=self.config['batch_size'], shuffle=True)
        
        # endregion
        
        # region Network.

        self.net = LxmInterpreter(self.lxc_size, self.num_blocks-1, self.uniform_len, **self.config['network']['hparams'])

        self.net.to(self.device)
        
        # endregion
        
        # region Optimizer.

        if self.config['optimizer']['name'] == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                              **self.config['optimizer']['hparams'])
        elif self.config['optimizer']['name'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(params=self.net.parameters(),
                                               **self.config['optimizer']['hparams'])
        else:
            raise NotImplementedError

        # endregion
        
        # region Criterion.
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # endregion
        
    def train(self) -> None:
        # region Start timing.

        since = time.time()

        # endregion
        
        # region Epoch loop.

        log = [{}]
        num_epoch = self.config["num_epoch"]
        for epoch in range(1, num_epoch+1):
            print(f'\nEpoch: {epoch}/{num_epoch}')

            self.net.train()
            
            loss_train = 0.
            acc_top1_train = 0.
            counter = 0
            
            # region Data loader loop.
            
            pbar = tqdm(total=len(self.dataset_train), ascii=True)
            for _, batch in enumerate(self.data_loader_train):
                # region Prepare data.
                
                aud = batch["audio"].to(self.device)  # [N, L, D].
                lxm_idx = batch["lexeme_index"].long().to(self.device)  # [N, B].

                N = aud.shape[0]
                
                # endregion
                
                # region Forward.
                
                self.optimizer.zero_grad()
                
                logits = self.net(aud[:, self.uniform_len:, :], lxm_idx[:, :-1])
                lxm_idx_pred = torch.argmax(logits, dim=-1)
                
                # endregion
                
                # region Loss and net weights update.

                loss = self.criterion(logits.reshape(-1, logits.shape[-1]),
                                      lxm_idx[:, 1:].reshape(-1))
                
                loss.backward()
                self.optimizer.step()
                
                loss_train += loss.item() * N
                lxm_idx = lxm_idx.detach().cpu().numpy()
                lxm_idx_pred = lxm_idx_pred.detach().cpu().numpy()
                acc_top1_train += cal_acc(lxm_idx_pred, lxm_idx[:, 1:]) * N
                counter += N
                
                # endregion
                
                # region Pbar update.

                pbar.set_description('lr: %s' % (str(self.optimizer.param_groups[0]['lr'])))
                pbar.update(N)
                
                # endregion
            
            pbar.close()
            
            # endregion
            
            # region Epoch loss and log.
            
            loss_train /= counter
            acc_top1_train /= counter
            
            print('Training',
                  f'Loss: {loss_train:.5f}',
                  f'Acc: {acc_top1_train:.5f}'
                  )

            self.writer.add_scalar(tag="Loss_TF/Train", scalar_value=loss_train, global_step=epoch)
            self.writer.add_scalar(tag="Acc_Top1_TF/Train", scalar_value=acc_top1_train, global_step=epoch)
            
            # endregion
            
            # region Checkpoints.

            if epoch % self.config['checkpoint_save_epoch_num'] == 0:
                torch.save(self.net.state_dict(),
                           os.path.join(self.log_dir, 'Checkpoints', f'checkpoint_{epoch // 1000}k{epoch % 1000}.pth'))
            
            # endregion

            # region Validation.
            
            valid_num_epoch = self.config['valid_num_epoch']
            if epoch % valid_num_epoch == 0:
                # write training log
                log[-1]['epoch'] = epoch
                log[-1]['loss'] = {'tf_train': f'{loss_train:.5f}'}
                log[-1]['acc'] = {'top1_tf_train': f'{acc_top1_train:.5f}'}
                log[-1]['lxm_sample'] = {'train': {'gt': [np.array2string(lxm_idx[i, 1:], separator=',') for i in range(3)],
                                                   'pred_tf': [np.array2string(lxm_idx_pred[i, :], separator=',') for i in range(3)]}}

                self.net.eval()
                
                loss_tf_valid, loss_valid = 0., 0.  # tf: teacher forcing
                acc_top1_tf_valid, acc_top1_valid = 0., 0.
                counter = 0
                
                with torch.no_grad():
                    for _, batch in enumerate(self.data_loader_val):
                        # region Prepare data.

                        aud = batch["audio"].to(self.device)  # [N, L, D].
                        lxm_idx = batch["lexeme_index"].long().to(self.device)  # [N, B].

                        N = aud.shape[0]

                        # endregion

                        # region Forward.

                        logits_tf = self.net(aud[:, self.uniform_len:, :], lxm_idx[:, :-1])
                        lxm_idx_pred_tf = torch.argmax(logits_tf, dim=-1)
                        lxm_idx_pred, logits = self.net.generate(aud[:, self.uniform_len:, :], lxm_idx[:, :1])

                        # endregion

                        loss_tf = self.criterion(logits_tf.reshape(-1, logits_tf.shape[-1]),
                                                 lxm_idx[:, 1:].reshape(-1))
                        loss = self.criterion(logits.reshape(-1, logits.shape[-1]),
                                              lxm_idx[:, 1:].reshape(-1))
                        
                        loss_tf_valid += loss_tf.item() * N
                        loss_valid += loss.item() * N
                        lxm_idx = lxm_idx.detach().cpu().numpy()
                        lxm_idx_pred = lxm_idx_pred.detach().cpu().numpy()
                        lxm_idx_pred_tf = lxm_idx_pred_tf.detach().cpu().numpy()
                        acc_top1_tf_valid += cal_acc(lxm_idx_pred_tf, lxm_idx[:, 1:]) * N
                        acc_top1_valid += cal_acc(lxm_idx_pred, lxm_idx[:, 1:]) * N
                        counter += N
                
                    loss_tf_valid /= counter
                    loss_valid /= counter
                    acc_top1_tf_valid /= counter
                    acc_top1_valid /= counter

                    print('Validation',
                          f'Loss_TF: {loss_tf_valid:.5f}',
                          f'Loss: {loss_valid:.5f}',
                          f'Acc_TF: {acc_top1_tf_valid:.5f}',
                          f'Acc: {acc_top1_valid:.5f}'
                          )

                    self.writer.add_scalar(tag="Loss/Valid", scalar_value=loss_valid, global_step=epoch)
                    self.writer.add_scalar(tag="Loss_TF/Valid", scalar_value=loss_tf_valid, global_step=epoch)
                    self.writer.add_scalar(tag="Acc_Top1/Valid", scalar_value=acc_top1_valid, global_step=epoch)
                    self.writer.add_scalar(tag="Acc_Top1_TF/Valid", scalar_value=acc_top1_tf_valid, global_step=epoch)

                    # write validation log
                    log[-1]['loss']['valid'] = f'{loss_valid:.5f}'
                    log[-1]['loss']['tf_valid'] = f'{loss_tf_valid:.5f}'
                    log[-1]['acc']['top1_valid'] = f'{acc_top1_valid:.5f}'
                    log[-1]['acc']['top1_tf_valid'] = f'{acc_top1_tf_valid:.5f}'
                    log[-1]['lxm_sample']['valid'] = {'gt': [np.array2string(lxm_idx[i, 1:], separator=',') for i in range(3)],
                                                      'pred': [np.array2string(lxm_idx_pred[i, :], separator=',') for i in range(3)],
                                                      'pred_tf': [np.array2string(lxm_idx_pred_tf[i, :], separator=',') for i in range(3)]}
                    with open(os.path.join(self.log_dir, 'log.json5'), 'w') as f:
                        json5.dump(log, f, indent=4)
                    log.append({})
            
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