# region Import.

import os
import sys
import torch
import shutil
import json5
import pickle
import argparse
import warnings

import joblib as jl
import cv2 as cv

from dataset import *
from utils import *
from sklearn.pipeline import Pipeline

module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if module_path not in sys.path:
    sys.path.append(module_path)

from Data_Preprocessing.Utils.preprocessing import *
from Data_Preprocessing.Utils.writers import BVHWriter
from Data_Preprocessing.Utils.parsers import BVHParser
from Data_Preprocessing.Utils.utils import inv_standardize, expmap_to_euler, resample_motion, split_dataset
from Data_Preprocessing.preprocess import Preprocessor
from Data_Preprocessing.Utils.Pymo.BVH_loader import load as bvh_load
from Data_Preprocessing.Utils.Pymo.BVH_loader import save as bvh_save

from Gesture_Lexicon.lexicon import predict_lexeme

# endregion


def get_args_parser():
    parser = argparse.ArgumentParser('gesture generator', add_help=False)

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--name_file', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--checkpoint_config', type=str, required=True)
    parser.add_argument('--lxc_checkpoint_path', type=str, required=True)
    parser.add_argument('--lxc_checkpoint_config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./Result')

    return parser


class Inference:
    def __init__(self,
                 path_pretrained: str, path_config: str,
                 path_lxc_net: str, path_lxc_config: str,
                 device: str = 'cuda:0') -> None:
        # region Load train config.

        with open(path_config, "r") as f:
            self.config = json5.load(f)

        # endregion

        # region Load data preprocessing config.

        with open(os.path.join(self.config["dir_data"], 'config.json5'), 'r') as f:
            self.config_data_preprocessing = json5.load(f)
        self.fps = self.config_data_preprocessing['fps']
        self.uniform_len = self.config_data_preprocessing["uniform_len"]
        self.num_blocks = self.config_data_preprocessing["num_blocks_per_clip"]
        self.rotation_order = self.config_data_preprocessing["bvh_rotation_order"]

        # endregion

        self.path_lxc_net = path_lxc_net
        self.path_lxc_config = path_lxc_config

        # region Device.

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # endregion

        # region Load pretrained net.

        self.net = initialize_net(self.config, self.config_data_preprocessing)

        self.net.load_state_dict(torch.load(path_pretrained, map_location=torch.device("cpu")))
        self.net.to(self.device)

        self.net.eval()

        # endregion

    def sample(self, dir_data: str, name_file: str, dir_save: str = "./"):
        # region Prepare preprocess config.

        dataset_type = "valid"

        os.makedirs(dir_save, exist_ok=True)

        config_preprocess = copy.deepcopy(self.config_data_preprocessing)

        dir_processed_dataset = copy.deepcopy(config_preprocess["dir_save"])
        config_preprocess["dir_data"] = dir_data
        config_preprocess["dir_save"] = dir_save
        config_preprocess["split_data"] = False

        with open(os.path.join(dir_save, "config.json5"), "w") as f:
            json5.dump(config_preprocess, f, indent=4)

        # endregion

        # region Data preprocessing.

        preprocessor = Preprocessor(os.path.join(dir_save, "config.json5"))
        preprocessor.preprocess()

        # endregion

        # region Get dataset.

        num_blocks = len(np.load(os.path.join(dir_save, "Features", "Data_Len_Uniform", name_file + "_index.npy")))
        uniform_len = self.uniform_len

        _, _, _ = split_dataset(dir_audio_feat=os.path.join(dir_save, "Features", "Audio_Feature"),
                                dir_motion_expmap=os.path.join(dir_save, "Features", "Motion_Expmap"),
                                dir_data_len_uniform=os.path.join(dir_save, "Features", "Data_Len_Uniform"),
                                names_file=[name_file], dir_save=dir_save,
                                uniform_len=uniform_len, num_blocks_per_clip=num_blocks, step=num_blocks,  # means batch size is 1
                                dataset_type=dataset_type, train_scaler_dir=dir_processed_dataset, save=True)

        # load lexicon
        with open(os.path.join(dir_processed_dataset, "lexicon.pkl"), "rb") as f:
            lexicon = pickle.load(f)
        lexicon_size = len(lexicon.cluster_centers_)

        predict_lexeme(os.path.join(dir_save, f"{dataset_type}.npz"),
                       self.path_lxc_net, self.path_lxc_config,
                       os.path.join(dir_processed_dataset, 'train_lexeme_scaler.sav'),
                       lexicon, self.device, True)

        dataset = TrainingDataset(os.path.join(dir_save, f"{dataset_type}.npz"))
        batch = dataset[:]
        for k in batch.keys():
            batch[k] = torch.from_numpy(batch[k])

        # endregion

        # region HARDCODE: Load DataFrame template.

        self.data_frame_template_expmap = pd.read_csv(os.path.join(dir_save, "Features", "Motion_Expmap", name_file + ".csv"),
                                                      index_col=0)
        self.data_frame_template_euler = pd.read_csv(os.path.join(dir_save, "Features", "BVH_Motion", name_file + ".csv"),
                                                     index_col=0)

        # endregion

        # region HARDCODE: Create data pipeline.
        
        data_pipe = jl.load(os.path.join(self.config["dir_data"], 'data_pipe.sav'))
        new_step = ('np', Numpyfier())
        data_pipe.steps.append(new_step)
        self.data_pipe = data_pipe
        
        path_bvh_example = os.path.join(dir_data, "Motion", name_file + ".bvh")
        parser = BVHParser()
        data_original = [parser.parse(os.path.join(path_bvh_example))]
        for step_name, step_transformer in self.data_pipe.steps[:-1]:
            data_original = step_transformer.transform(data_original)
        last_step_name, last_transformer = self.data_pipe.steps[-1]
        _ = last_transformer.fit_transform(data_original)

        # endregion

        # region Infer.

        batch_size = batch["audio"].shape[0]  # N is 1

        infer_res = infer_train(batch, self.device, self.net, uniform_len, num_blocks, self.config["network"]["name"])

        motion_gt = infer_res[0].detach().cpu().numpy()  # [N, L, D]
        motion_pred = infer_res[1].detach().cpu().numpy()

        assert motion_gt.shape[1] == motion_pred.shape[1]
        assert motion_gt.shape[1] == (num_blocks - 2) * uniform_len

        motion_gt_tmp = batch["motion"].numpy()
        motion_gt = np.concatenate([motion_gt_tmp[:, : uniform_len, :], motion_gt, motion_gt_tmp[:, -uniform_len:, :]],
                                   axis=1)
        motion_pred = np.concatenate(
            [motion_gt_tmp[:, : uniform_len, :], motion_pred, motion_gt_tmp[:, -uniform_len:, :]], axis=1)

        assert motion_gt.shape[1] == motion_pred.shape[1]
        assert motion_gt.shape[1] == num_blocks * uniform_len

        # endregion

        # region Inverse scaler.

        motion_scaler = jl.load(os.path.join(dir_processed_dataset, "train_motion_scaler.sav"))

        motion_gt = inv_standardize(motion_gt, motion_scaler)
        motion_pred = inv_standardize(motion_pred, motion_scaler)

        # endregion

        # region Expmap to euler.

        motion_gt_euler = []
        motion_pred_euler = []
        for i in range(batch_size):
            data_frame_gt = pd.DataFrame(data=motion_gt[i],
                                         columns=self.data_frame_template_expmap.columns,
                                         index=pd.to_timedelta(np.arange(motion_gt[i].shape[0]) / self.fps, unit='s'))
            motion_gt_euler.append(expmap_to_euler(data_frame_gt, self.rotation_order).to_numpy())

            data_frame_pred = pd.DataFrame(data=motion_pred[i],
                                           columns=self.data_frame_template_expmap.columns,
                                           index=pd.to_timedelta(np.arange(motion_pred[i].shape[0]) / self.fps,
                                                                 unit='s'))
            motion_pred_euler.append(expmap_to_euler(data_frame_pred, self.rotation_order).to_numpy())

        # endregion

        # region Write clip uniformed as bvh.

        os.makedirs(dir_save, exist_ok=True)

        writer = BVHWriter()

        # endregion

        # region Construct index-scope pair.

        index_scope_pair = {}

        idx_tmp = np.load(os.path.join(dir_save, "Features", "Data_Len_Uniform", name_file + "_index.npy")).astype(int).reshape(-1)
        scope_tmp = np.load(os.path.join(dir_save, "Features", "Data_Len_Uniform", name_file + "_scope.npy")).astype(int)

        assert idx_tmp.shape[0] == scope_tmp.shape[0]

        for cc, idx in enumerate(idx_tmp):
            if idx not in index_scope_pair.keys():
                index_scope_pair[idx] = scope_tmp[cc]
            else:
                raise ValueError

        # endregion

        # region Construct index-motion pair.

        indexes_selected = batch["index"].numpy().astype(int)  # (num_clips, num_blocks).

        index_motion_gt_pair = {}
        index_motion_pred_pair = {}
        for row in range(indexes_selected.shape[0]):
            for col in range(indexes_selected.shape[1]):
                key = indexes_selected[row][col]

                if (key not in index_motion_gt_pair.keys()) and (key not in index_motion_pred_pair.keys()):
                    if col == 0:
                        index_motion_gt_pair[key] = motion_gt_euler[row][col * self.uniform_len: (col + 1) * self.uniform_len, :]
                        index_motion_pred_pair[key] = motion_pred_euler[row][col * self.uniform_len: (col + 1) * self.uniform_len, :]
                    else:
                        index_motion_gt_pair[key] = motion_gt_euler[row][col * self.uniform_len - 1: (col + 1) * self.uniform_len, :]
                        index_motion_pred_pair[key] = motion_pred_euler[row][col * self.uniform_len - 1: (col + 1) * self.uniform_len, :]
                else:
                    raise ValueError

        # endregion

        # region Resample motion.

        motions_gt_resampled = []
        motions_pred_resampled = []
        for row in range(indexes_selected.shape[0]):
            motions_gt_resampled_tmp = []
            motions_pred_resampled_tmp = []

            for col in range(indexes_selected.shape[1]):
                idx_tmp = indexes_selected[row][col]
                scope_tmp = index_scope_pair[idx_tmp]
                data_frame_gt = pd.DataFrame(data=index_motion_gt_pair[idx_tmp],
                                             columns=self.data_frame_template_euler.columns,
                                             index=pd.to_timedelta(
                                                 np.arange(index_motion_gt_pair[idx_tmp].shape[0]) / self.fps,
                                                 unit='s'))
                data_frame_pred = pd.DataFrame(data=index_motion_pred_pair[idx_tmp],
                                               columns=self.data_frame_template_euler.columns,
                                               index=pd.to_timedelta(
                                                   np.arange(index_motion_pred_pair[idx_tmp].shape[0]) / self.fps,
                                                   unit='s'))

                if col == 0:
                    ticks = np.arange(self.uniform_len)
                    target_ticks = (np.arange(scope_tmp[0], scope_tmp[1] + 1) - scope_tmp[0]) * (
                            (self.uniform_len - 1) / (scope_tmp[1] - scope_tmp[0]))
                    target_ticks[-1] = ticks[-1]

                    motions_gt_resampled_tmp.append(
                        resample_motion(data_frame_gt, ticks, target_ticks, self.rotation_order).to_numpy())
                    motions_pred_resampled_tmp.append(
                        resample_motion(data_frame_pred, ticks, target_ticks, self.rotation_order).to_numpy())
                else:
                    ticks = np.arange(self.uniform_len + 1)
                    target_ticks = (np.arange(scope_tmp[0], scope_tmp[1] + 1) - scope_tmp[0]) * (
                            (self.uniform_len) / (scope_tmp[1] - scope_tmp[0]))
                    target_ticks[-1] = ticks[-1]

                    motions_gt_resampled_tmp.append(
                        resample_motion(data_frame_gt, ticks, target_ticks, self.rotation_order).to_numpy()[1:, :])
                    motions_pred_resampled_tmp.append(
                        resample_motion(data_frame_pred, ticks, target_ticks, self.rotation_order).to_numpy()[1:, :])

            motions_gt_resampled.append(np.concatenate(motions_gt_resampled_tmp))
            motions_pred_resampled.append(np.concatenate(motions_pred_resampled_tmp))

        # endregion

        # region Gaussian filtering.

        kernel = cv.getGaussianKernel(5, 1)
        for i in range(batch_size):
            motions_pred_resampled[i] = cv.filter2D(motions_pred_resampled[i], -1, kernel=kernel)

        # endregion

        # region Write clip unwarped as bvh.

        for i in range(batch_size):
            anim_clips = [motions_gt_resampled[i], motions_pred_resampled[i]]
            inv_clips = self.data_pipe.inverse_transform(anim_clips)

            with open(os.path.join(dir_save, f"{name_file}_gt.bvh"), "w") as f:
                writer.write(inv_clips[0], f, framerate=self.fps)

            with open(os.path.join(dir_save, f"{name_file}_pred.bvh"), "w") as f:
                writer.write(inv_clips[1], f, framerate=self.fps)

        # endregion

        # region Transfer bvh motion to facing coordinate.

        for i in range(batch_size):
            path_bvh = [
                os.path.join(dir_save, f"{name_file}_gt.bvh"),
                os.path.join(dir_save, f"{name_file}_pred.bvh")
            ]

            for p in path_bvh:
                motion_data = bvh_load(p)
                motion_data = motion_data.to_facing_coordinate()

                bvh_save(motion_data, p, euler_order=self.rotation_order)

        # endregion


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    
    # region Data Preparation.
    
    if not os.path.exists(os.path.join(args.data_dir, args.name_file+'.wav')):
        raise ValueError(f"Audio file {args.name_file+'.wav'} does not exist in {args.data_dir}.")
    
    if not os.path.exists(os.path.join(args.data_dir, args.name_file+'.bvh')):
        raise ValueError(f"Motion file {args.name_file+'.bvh'} does not exist in {args.data_dir}.")
    
    for f_name in os.listdir(args.data_dir):
        if ('.wav' in f_name) or ('.bvh' in f_name):
            if f_name[:-4] != args.name_file:
                warnings.warn('There are other audio or motion files in the data directory. Will remove it.')
                os.remove(os.path.join(args.data_dir, f_name))
    
    os.makedirs(os.path.join(args.data_dir, 'Audio'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'Motion'), exist_ok=True)
    
    os.rename(os.path.join(args.data_dir, args.name_file+'.wav'), os.path.join(args.data_dir, 'Audio', args.name_file+'.wav'))
    os.rename(os.path.join(args.data_dir, args.name_file+'.bvh'), os.path.join(args.data_dir, 'Motion', args.name_file+'.bvh'))
    
    if os.path.exists(os.path.join(args.data_dir, 'Features')):
        shutil.rmtree(os.path.join(args.data_dir, 'Features'))
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # endregion

    inf = Inference(args.checkpoint_path, args.checkpoint_config,
                    args.lxc_checkpoint_path, args.lxc_checkpoint_config,
                    args.device)

    inf.sample(args.data_dir, args.name_file, args.save_dir)
    
    # region Result Preparation.
    
    os.rename(os.path.join(args.data_dir, 'Audio', args.name_file+'.wav'), os.path.join(args.data_dir, args.name_file+'.wav'))
    os.rename(os.path.join(args.data_dir, 'Motion', args.name_file+'.bvh'), os.path.join(args.data_dir, args.name_file+'.bvh'))
    
    shutil.rmtree(os.path.join(args.data_dir, 'Audio'))
    shutil.rmtree(os.path.join(args.data_dir, 'Motion'))
    shutil.rmtree(os.path.join(args.data_dir, 'Features'))
    os.remove(os.path.join(args.data_dir, 'config.json5'))
    os.remove(os.path.join(args.data_dir, 'data_pipe.sav'))
    os.remove(os.path.join(args.data_dir, 'motion_template.bvh'))
    os.remove(os.path.join(args.data_dir, 'valid_info.txt'))
    os.remove(os.path.join(args.data_dir, 'valid.npz'))
    
    print('\n\n------------------------------------------------------------------')
    print(f'The generated motion is saved in {os.path.join(args.save_dir, args.name_file+"_pred.bvh")}.')
    print(f'The GT motion is saved in {os.path.join(args.save_dir, args.name_file+"_gt.bvh")}.')
    
    # endregion
            