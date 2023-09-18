# region Import.

import os
import librosa

import numpy as np
import pandas as pd
import joblib as jl

from .rotation_tools import euler2expmap, unroll, expmap2euler
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

# endregion


__all__ = ["nan_smooth", "import_beat_label", "resample_motion", "resample_audio_feature", "resample_wav_data",
           "euler_to_expmap", "split_dataset", "inv_standardize", "expmap_to_euler"]


def nan_smooth(data, filt_len):
    win = filt_len // 2

    nframes = len(data)
    out = np.zeros(data.shape)
    for i in range(nframes):
        if i < win:
            st = 0
        else:
            st = i - win
        if i > nframes - win - 1:
            en = nframes
        else:
            en = i + win + 1
        out[i, :] = np.nanmean(data[st:en, :], axis=0)
    return out


def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler


def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled


def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled


def import_beat_label(path_beat : str):
    with open(path_beat, 'r') as f:
        beats = f.readlines()

    beat_times = []
    for b in beats:
        if 's' in b:
            beat_times.append(float(b.split(' ')[-1][0: -2]))
    
    return np.array(beat_times)


def resample_motion(motion : pd.DataFrame, ticks : np.ndarray, target_ticks : np.ndarray, 
                    rotation_order : str) -> pd.DataFrame:
    """
    The unit of ticks: s.
    """
    
    new_motion = pd.DataFrame(index=pd.to_timedelta(target_ticks, unit='s'))
    
    joint_infos = {}
    for col in motion.columns:
        infos = col.split('_')
        channel = infos[-1][1:]
        joint = '_'.join(infos[:-1])
        
        if joint not in joint_infos.keys():
            joint_infos[joint] = [channel]
        else:
            joint_infos[joint].append(channel)
    
    for joint, channels in joint_infos.items():
        # Deal with position using linear interp.
        if 'position' in channels:
            positions = np.array(motion[['%s_Xposition' % joint,
                                         '%s_Yposition' % joint,
                                         '%s_Zposition' % joint]])
            positions = interp1d(ticks, positions,
                                 kind='linear',
                                 axis=0, copy=False, bounds_error=True, assume_sorted=True)(target_ticks)
            new_motion[['%s_Xposition' % joint,
                        '%s_Yposition' % joint,
                        '%s_Zposition' % joint]] = pd.DataFrame(data=positions,
                                                                index=new_motion.index)
        
        # Deal with rotations using slerp.
        if 'rotation' in channels:
            rotations_euler = np.array(motion[['%s_%srotation' % (joint, rotation_order[0]),
                                               '%s_%srotation' % (joint, rotation_order[1]),
                                               '%s_%srotation' % (joint, rotation_order[2])]])
            rotations = Rotation.from_euler(
                rotation_order.upper(), rotations_euler, degrees=True)
            rotations = Slerp(ticks, rotations)(target_ticks).as_euler(
                rotation_order.upper(), degrees=True)
            new_motion[['%s_%srotation' % (joint, rotation_order[0]),
                        '%s_%srotation' % (joint, rotation_order[1]),
                        '%s_%srotation' % (joint, rotation_order[2])]] = pd.DataFrame(data=rotations,
                                                                                      index=new_motion.index)

        if ('position' not in channels) and ('rotation' not in channels):
            print(channels)
            raise ValueError('Motion channel is wrong.')
    
    return new_motion


def resample_audio_feature(audio_feat : np.ndarray, ticks : np.ndarray, target_ticks : np.ndarray) -> np.ndarray:
    return interp1d(ticks, audio_feat,
                    kind='linear',
                    axis=0, copy=False, bounds_error=True, assume_sorted=True)(target_ticks)


def resample_wav_data(wav_data: np.ndarray, target_len: int) -> np.ndarray:
    ratio = wav_data.shape[0] / target_len
    wav_data_resampled = librosa.effects.time_stretch(wav_data, ratio)

    if wav_data_resampled.shape[0] > target_len:
        wav_data_resampled = wav_data_resampled[: target_len]
    elif wav_data_resampled.shape[0] < target_len:
        wav_data_tmp = np.zeros(target_len)
        wav_data_tmp[: wav_data_resampled.shape[0]] = wav_data_resampled[:]
        wav_data_tmp[wav_data_resampled.shape[0]:] = wav_data_resampled[-1]
        wav_data_resampled = wav_data_tmp

    assert wav_data_resampled.shape[0] == target_len

    return wav_data_resampled


def euler_to_expmap(motion : pd.DataFrame, rotation_order : str) -> pd.DataFrame:
    new_motion = pd.DataFrame(index=motion.index)
    
    joint_infos = {}
    for col in motion.columns:
        infos = col.split('_')
        channel = infos[-1][1:]
        joint = '_'.join(infos[:-1])
        
        if joint not in joint_infos.keys():
            joint_infos[joint] = [channel]
        else:
            joint_infos[joint].append(channel)
    
    for joint, channels in joint_infos.items():
        if "rotation" in channels:
            r_cols = [
                "%s_%srotation" % (joint, rotation_order[0]), 
                "%s_%srotation" % (joint, rotation_order[1]), 
                "%s_%srotation" % (joint, rotation_order[2])
            ]
            r = motion[r_cols].to_numpy()
            
            exps = unroll(np.array([euler2expmap(f, rotation_order, True) for f in r]))
            
            new_motion[["%s_gamma" % joint, 
                        "%s_beta" % joint, 
                        "%s_alpha" % joint]] = pd.DataFrame(data=exps[:, [2, 1, 0]], index=new_motion.index)
        
        if "position" in channels:
            p_cols = [
                "%s_Xposition" % joint,
                "%s_Yposition" % joint,
                "%s_Zposition" % joint
            ]
            
            new_motion[p_cols] = motion[p_cols].copy()

        if ('position' not in channels) and ('rotation' not in channels):
            print(channels)
            raise ValueError('Motion channel is wrong.')
    
    assert len(motion.columns) == len(new_motion.columns)
    
    return new_motion


def expmap_to_euler(motion : pd.DataFrame, rotation_order : str) -> pd.DataFrame:
    new_motion = pd.DataFrame(index=motion.index)
    
    joint_infos = {}
    for col in motion.columns:
        infos = col.split('_')
        if ("position" in infos[-1]) or ("rotation" in infos[-1]):
            channel = infos[-1][1:]
        else:
            channel = infos[-1]
        joint = '_'.join(infos[:-1])
        
        if joint not in joint_infos.keys():
            joint_infos[joint] = [channel]
        else:
            joint_infos[joint].append(channel)
    
    for joint, channels in joint_infos.items():
        if ("alpha" in channels) and ("beta" in channels) and ("gamma" in channels):
            r_cols = [
                "%s_alpha" % joint, 
                "%s_beta" % joint, 
                "%s_gamma" % joint
            ]
            r = motion[r_cols].to_numpy()
            
            euler = np.array([expmap2euler(f, rotation_order, True) for f in r])
            
            new_motion[["%s_%srotation" % (joint, rotation_order[0]), 
                        "%s_%srotation" % (joint, rotation_order[1]), 
                        "%s_%srotation" % (joint, rotation_order[2])]] = pd.DataFrame(data=euler[:, :], index=new_motion.index)
        
        if "position" in channels:
            p_cols = [
                "%s_Xposition" % joint,
                "%s_Yposition" % joint,
                "%s_Zposition" % joint
            ]
            
            new_motion[p_cols] = motion[p_cols].copy()

        if ('position' not in channels) and ('alpha' not in channels):
            print(channels)
            raise ValueError('Motion channel is wrong.')
    
    assert len(motion.columns) == len(new_motion.columns)
    
    return new_motion


def split_dataset(dir_audio_feat: str, dir_motion_expmap: str, dir_data_len_uniform: str,
                  names_file: List[str], dir_save: str,
                  uniform_len: int, num_blocks_per_clip: int, step: int,
                  dataset_type: str = "train", train_scaler_dir=None, save: bool = True):
    res_mel = []
    res_motion = []
    res_index = []
    
    def process(suffix : str = ""):
        for n in names_file:
            index = np.load(os.path.join(dir_data_len_uniform, n + f"_index{suffix}.npy")).astype(int)
            mel = np.load(os.path.join(dir_audio_feat, n + f"{suffix}.npy"))
            motion = pd.read_csv(os.path.join(dir_motion_expmap, n + f"{suffix}.csv"), index_col=0).to_numpy()
            
            assert mel.shape[0] == motion.shape[0]
            assert mel.shape[0] % uniform_len == 0
            assert index.shape[0] == int(mel.shape[0] / uniform_len)
            
            index_clips = []
            mel_clips = []
            motion_clips = []
            step_frame = int(uniform_len * step)
            for i in range(0, mel.shape[0], step_frame):
                if (i + uniform_len * num_blocks_per_clip) <= mel.shape[0]:
                    s = i
                    e = i + uniform_len * num_blocks_per_clip
                    mel_clips.append(mel[s: e, :])
                    motion_clips.append(motion[s: e, :])
                    index_clips.append(index[int(s / uniform_len): int(e / uniform_len)])

            if len(mel_clips) > 0:
                res_mel.append(np.array(mel_clips))
                res_motion.append(np.array(motion_clips))
                res_index.append(np.array(index_clips))

                print(n + f"_mel{suffix}:", res_mel[-1].shape)
                print(n + f"_motion{suffix}:", res_motion[-1].shape)
    
    process()
    
    res_mel = np.concatenate(res_mel, axis=0)
    res_motion = np.concatenate(res_motion, axis=0)
    res_index = np.concatenate(res_index, axis=0).astype(int)  # num_clips X num_blocks.
    
    assert res_mel.shape[0] == res_motion.shape[0]  # num_clips X time X dim_feat.
    assert res_index.shape[0] == res_mel.shape[0]

    if dataset_type == 'train':
        res_mel_normalized, audio_scaler = fit_and_standardize(res_mel)
        res_motion_normalized, motion_scaler = fit_and_standardize(res_motion)
    else:
        assert train_scaler_dir is not None
        audio_scaler = jl.load(os.path.join(train_scaler_dir, 'train_audio_scaler.sav'))
        motion_scaler = jl.load(os.path.join(train_scaler_dir, 'train_motion_scaler.sav'))
        res_mel_normalized = standardize(res_mel, audio_scaler)
        res_motion_normalized = standardize(res_motion, motion_scaler)
    
    if save:
        np.savez(os.path.join(dir_save, dataset_type + ".npz"),
                 audio=res_mel_normalized,
                 motion=res_motion_normalized,
                 index=res_index)
        if dataset_type == 'train':
            jl.dump(audio_scaler, os.path.join(dir_save, dataset_type + "_audio_scaler.sav"))
            jl.dump(motion_scaler, os.path.join(dir_save, dataset_type + "_motion_scaler.sav"))

        with open(os.path.join(dir_save, dataset_type + "_info.txt"), "w") as f:
            f.write("Feature Name | (Num_clips, Time/Num_blocks, Dim_feat):\n")
            f.write(f"Audio: {res_mel_normalized.shape}\n")
            f.write(f"Motion: {res_motion_normalized.shape}\n")
            f.write(f"Index: {res_index.shape}\n")
    
    print(f"Audio {dataset_type}:", res_mel_normalized.shape)
    print(f"Motion {dataset_type}:", res_motion_normalized.shape)
    print(f"Index {dataset_type}:", res_index.shape)
    
    return res_mel_normalized, res_motion_normalized, res_index