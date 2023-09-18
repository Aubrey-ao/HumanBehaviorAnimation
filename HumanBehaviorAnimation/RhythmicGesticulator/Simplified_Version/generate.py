import json5
import torch
import pickle
import shutil
import argparse
import warnings

import joblib as jl
import cv2 as cv

from utils import *
from scipy import stats
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from Data_Preprocessing.Utils.preprocessing import *
from Data_Preprocessing.Utils.writers import BVHWriter
from Data_Preprocessing.Utils.parsers import BVHParser
from Data_Preprocessing.preprocess import Preprocessor
from Data_Preprocessing.Utils.utils import inv_standardize, expmap_to_euler, resample_motion, split_dataset


def get_args_parser():
    parser = argparse.ArgumentParser('generate gesture', add_help=False)

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--name_file', type=str, required=True)
    parser.add_argument('--gen_checkpoint_path', type=str, required=True)
    parser.add_argument('--gen_checkpoint_config', type=str, required=True)
    parser.add_argument('--lxm_intp_checkpoint_path', type=str, required=True)
    parser.add_argument('--lxm_intp_checkpoint_config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./Result')

    return parser


class Inference:
    def __init__(self, gen_ckpt_path, gen_cfg_path,
                 lxm_intp_ckpt_path, lxm_intp_cfg_path,
                 device='cuda:0'):
        with open(gen_cfg_path, 'r') as f:
            self.gen_cfg = OmegaConf.create(json5.load(f))

        with open(lxm_intp_cfg_path, 'r') as f:
            self.lxm_intp_cfg = OmegaConf.create(json5.load(f))

        split_path = self.gen_cfg.dir_data.split(os.path.sep)
        self.dataset_dir = os.path.join('.', *split_path[split_path.index('Data'):])
        with open(os.path.join(self.dataset_dir, 'config.json5'), 'r') as f:
            self.data_cfg = OmegaConf.create(json5.load(f))
        self.fps = self.data_cfg.fps
        self.rotation_order = self.data_cfg["bvh_rotation_order"]

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.gen, self.lxm_intp = initialize_net(self.data_cfg, self.gen_cfg, self.lxm_intp_cfg)
        self.gen.load_state_dict(torch.load(gen_ckpt_path, map_location=torch.device("cpu")))
        self.gen.to(self.device)
        self.gen.eval()
        self.lxm_intp.load_state_dict(torch.load(lxm_intp_ckpt_path, map_location=torch.device("cpu")))
        self.lxm_intp.to(self.device)
        self.lxm_intp.eval()

    def sample(self, dir_data, name_file, dir_save="./"):
        # preprocess data
        dataset_type = "valid"

        os.makedirs(os.path.join(dir_data, 'Audio'), exist_ok=True)
        shutil.move(os.path.join(dir_data, f'{name_file}.wav'), os.path.join(dir_data, 'Audio'))
        os.makedirs(os.path.join(dir_data, 'Motion'), exist_ok=True)
        align_motion_template_len_to_audio(aud_path=os.path.join(dir_data, 'Audio', f'{name_file}.wav'), 
                                           mo_path=os.path.join(self.dataset_dir, 'motion_template.bvh'),
                                           rotation_order=self.rotation_order,
                                           save_path=os.path.join(dir_data, 'Motion', f'{name_file}.bvh'))
        os.makedirs(dir_save, exist_ok=True)

        data_cfg = copy.deepcopy(self.data_cfg)
        data_cfg.dir_data = dir_data
        data_cfg.dir_save = dir_save
        data_cfg.split_data = False
        with open(os.path.join(dir_save, "config.json5"), "w") as f:
            json5.dump(data_cfg, f, indent=4)

        preprocessor = Preprocessor(os.path.join(dir_save, "config.json5"))
        preprocessor.preprocess()

        total_B = len(np.load(os.path.join(dir_save, "Features", "Data_Len_Uniform", name_file + "_index.npy")))
        split_dataset(dir_audio_feat=os.path.join(dir_save, "Features", "Audio_Feature"),
                      dir_motion_expmap=os.path.join(dir_save, "Features", "Motion_Expmap"),
                      dir_data_len_uniform=os.path.join(dir_save, "Features", "Data_Len_Uniform"),
                      names_file=[name_file], dir_save=dir_save,
                      uniform_len=data_cfg.uniform_len,
                      num_blocks_per_clip=total_B, step=total_B,  # means batch size is 1
                      dataset_type=dataset_type, train_scaler_dir=self.dataset_dir, save=True)

        with open(os.path.join(self.dataset_dir, "lexicon.pkl"), "rb") as f:
            lexicon = pickle.load(f)
        init_lxm_idx = stats.mode(lexicon.labels_).mode[0]  # default to the most frequent lexeme

        processed_data = np.load(os.path.join(dir_save, f"{dataset_type}.npz"))
        aud = processed_data['audio'].astype(np.float32)  # [N(1), L, D]
        mo = processed_data['motion'].astype(np.float32)  # [N(1), L, D]
        idx = processed_data['index'].astype(int)  # [N(1), B]

        self.data_frame_template_expmap = pd.read_csv(
            os.path.join(dir_save, "Features", "Motion_Expmap", name_file + ".csv"),
            index_col=0)
        self.data_frame_template_euler = pd.read_csv(
            os.path.join(dir_save, "Features", "BVH_Motion", name_file + ".csv"),
            index_col=0)
        
        data_pipe = jl.load(os.path.join(self.dataset_dir, 'data_pipe.sav'))
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

        # predict lexeme
        with torch.no_grad():
            BL = self.data_cfg.uniform_len
            aud = torch.from_numpy(aud).to(self.device)
            init_lxm_idx = torch.tensor([[init_lxm_idx]]).to(self.device).long()
            lxc = torch.from_numpy(lexicon.cluster_centers_).to(self.device)

            lxm_idx_pred = self.lxm_intp.generate(aud[:, BL:, :], init_lxm_idx)[0].long()
            lxm_idx_pred = torch.cat([init_lxm_idx, lxm_idx_pred], dim=-1)
            lxm_pred = lxc[lxm_idx_pred, :]
            
            # print(lxm_idx_pred)

        # generate motion
        with torch.no_grad():
            init_mo = torch.from_numpy(mo).to(self.device)[:, :BL, :]

            mo_hat = self.gen(aud.permute((0, 2, 1)).contiguous(),
                            init_mo.permute((0, 2, 1)).contiguous(),
                            lxm_pred.permute((0, 2, 1)).contiguous()).permute((0, 2, 1)).contiguous().cpu().numpy()
            mo_hat = np.concatenate([mo[:, :BL, :], mo_hat, mo[:, -BL:, :]], axis=1)
            assert mo_hat.shape[1] == total_B * BL

        # inverse scaler
        motion_scaler = jl.load(os.path.join(self.dataset_dir, "train_motion_scaler.sav"))
        motion_pred = inv_standardize(mo_hat, motion_scaler)

        # expmap to euler
        motion_pred_euler = []
        data_frame_pred = pd.DataFrame(data=motion_pred[0],
                                       columns=self.data_frame_template_expmap.columns,
                                       index=pd.to_timedelta(np.arange(motion_pred[0].shape[0]) / self.fps, unit='s'))
        motion_pred_euler.append(expmap_to_euler(data_frame_pred, self.rotation_order).to_numpy())

        # construct index-scope pair
        index_scope_pair = {}

        idx_tmp = np.load(os.path.join(dir_save, "Features", "Data_Len_Uniform", name_file + "_index.npy")).astype(
            int).reshape(-1)
        scope_tmp = np.load(os.path.join(dir_save, "Features", "Data_Len_Uniform", name_file + "_scope.npy")).astype(
            int)
        assert idx_tmp.shape[0] == scope_tmp.shape[0]

        for cc, _idx in enumerate(idx_tmp):
            if _idx not in index_scope_pair.keys():
                index_scope_pair[_idx] = scope_tmp[cc]
            else:
                raise ValueError

        # construct index-motion pair
        indexes_selected = idx.astype(int)  # (num_clips, num_blocks).
        index_motion_pred_pair = {}
        for row in range(indexes_selected.shape[0]):
            for col in range(indexes_selected.shape[1]):
                key = indexes_selected[row][col]
                if key not in index_motion_pred_pair.keys():
                    if col == 0:
                        index_motion_pred_pair[key] = motion_pred_euler[row][col * BL: (col + 1) * BL, :]
                    else:
                        index_motion_pred_pair[key] = motion_pred_euler[row][col * BL - 1: (col + 1) * BL, :]
                else:
                    raise ValueError

        # resample motion
        motions_pred_resampled = []
        for row in range(indexes_selected.shape[0]):
            motions_pred_resampled_tmp = []
            for col in range(indexes_selected.shape[1]):
                idx_tmp = indexes_selected[row][col]
                scope_tmp = index_scope_pair[idx_tmp]
                data_frame_pred = pd.DataFrame(data=index_motion_pred_pair[idx_tmp],
                                               columns=self.data_frame_template_euler.columns,
                                               index=pd.to_timedelta(
                                                   np.arange(index_motion_pred_pair[idx_tmp].shape[0]) / self.fps,
                                                   unit='s'))
                if col == 0:
                    ticks = np.arange(BL)
                    target_ticks = (np.arange(scope_tmp[0], scope_tmp[1] + 1) - scope_tmp[0]) * (
                            (BL - 1) / (scope_tmp[1] - scope_tmp[0]))
                    target_ticks[-1] = ticks[-1]
                    motions_pred_resampled_tmp.append(
                        resample_motion(data_frame_pred, ticks, target_ticks, self.rotation_order).to_numpy())
                else:
                    ticks = np.arange(BL + 1)
                    target_ticks = (np.arange(scope_tmp[0], scope_tmp[1] + 1) - scope_tmp[0]) * (
                            BL / (scope_tmp[1] - scope_tmp[0]))
                    target_ticks[-1] = ticks[-1]
                    motions_pred_resampled_tmp.append(
                        resample_motion(data_frame_pred, ticks, target_ticks, self.rotation_order).to_numpy()[1:, :])
            motions_pred_resampled.append(np.concatenate(motions_pred_resampled_tmp))

        # gaussian filtering
        kernel = cv.getGaussianKernel(5, 1)
        motions_pred_resampled[0] = cv.filter2D(motions_pred_resampled[0], -1, kernel=kernel)

        # write clip unwarped as bvh
        writer = BVHWriter()
        anim_clips = [motions_pred_resampled[0]]
        inv_clips = self.data_pipe.inverse_transform(anim_clips)
        with open(os.path.join(dir_save, f"{name_file}_pred.bvh"), "w") as f:
            writer.write(inv_clips[0], f, framerate=self.fps)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    
    # region Data Preparation.
    
    if not os.path.exists(os.path.join(args.data_dir, args.name_file+'.wav')):
        raise ValueError(f"Audio file {args.name_file+'.wav'} does not exist in {args.data_dir}.")
    
    for f_name in os.listdir(args.data_dir):
        if '.wav' in f_name:
            if f_name[:-4] != args.name_file:
                warnings.warn('There are other audio files in the data directory. Will remove it.')
                os.remove(os.path.join(args.data_dir, f_name))
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # endregion

    inf = Inference(args.gen_checkpoint_path, args.gen_checkpoint_config,
                    args.lxm_intp_checkpoint_path, args.lxm_intp_checkpoint_config,
                    args.device)

    inf.sample(args.data_dir, args.name_file, args.save_dir)
    
    # region Result Preparation.
    
    os.rename(os.path.join(args.data_dir, 'Audio', args.name_file+'.wav'), os.path.join(args.data_dir, args.name_file+'.wav'))
    
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
    
    # endregion
