# region Import.

import os

import librosa.feature
import numpy as np
import pandas as pd

from typing import List, Tuple
from .utils import *

# endregion


__all__ = ["align_mel_with_motion", "uniform_data_fragment_length_tsm"]


def align_mel_with_motion(dir_mel: str, dir_motion: str, names_file: List[str],
                           dir_save: str, save: bool = False) -> Tuple[List[np.ndarray], List[pd.DataFrame]]:
    res_mel = []
    res_motion = []

    def process(suffix: str = ""):
        for n in names_file:
            mel = np.load(os.path.join(dir_mel, n + ".npy"))
            motion = pd.read_csv(os.path.join(dir_motion, n + f"{suffix}.csv"), index_col=0)

            if motion.shape[0] > mel.shape[0]:
                motion = motion.drop(motion.index[mel.shape[0] - motion.shape[0]:])
            elif motion.shape[0] < mel.shape[0]:
                mel = mel[: motion.shape[0], :]

            assert motion.shape[0] == mel.shape[0]

            if save:
                np.save(os.path.join(dir_save, n + f"_mel{suffix}.npy"), mel)
                motion.to_csv(os.path.join(dir_save, n + f"_motion{suffix}.csv"))

            res_mel.append(mel)
            res_motion.append(motion)

            print(n + f'_mel{suffix}:', res_mel[-1].shape)
            print(n + f'_motion{suffix}:', res_motion[-1].shape)

    process()

    return res_mel, res_motion


def uniform_data_fragment_length_tsm(dir_mel_motion_aligned: str, dir_onset: str, dir_wav: str,
                                     names_file: List[str], dir_save: str,
                                     fps: int, sr: int, rotation_order: str, uniform_len: int,
                                     save: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[pd.DataFrame]]:

    res_mel = []
    res_motion = []
    res_scopes = []

    assert sr % fps == 0
    hop_len = sr // fps

    counter = 0

    def process(suffix: str = "", counter: int = 0) -> int:
        for n in names_file:
            mel = np.load(os.path.join(dir_mel_motion_aligned, n + f"_mel{suffix}.npy"))
            wav = np.load(os.path.join(dir_wav, n + ".npy"))
            motion = pd.read_csv(os.path.join(dir_mel_motion_aligned, n + f"_motion{suffix}.csv"), index_col=0)
            onset = np.load(os.path.join(dir_onset, n + ".npy")).reshape(-1).astype(int)

            wav_len_expected = motion.shape[0] * hop_len
            if wav.shape[0] > wav_len_expected:
                wav = wav[: wav_len_expected]
            elif wav.shape[0] < wav_len_expected:
                wav_new = np.zeros(wav_len_expected)
                wav_new[: wav.shape[0]] = wav[:]
                wav_new[wav.shape[0]:] = wav[-1]
                wav = wav_new

            onset_filtered = [o for o in onset if 0 < o < (motion.shape[0] - 1)]

            scopes = []
            for i, o in enumerate(onset_filtered):
                if i == 0:
                    scopes.append((0, o))
                else:
                    scopes.append((onset_filtered[i - 1], o))
            scopes.append((onset_filtered[-1], motion.shape[0] - 1))

            wavs_resampled = []
            motions_resampled = []
            for idx, s in enumerate(scopes):
                if idx == 0:
                    ticks = (np.arange(s[0], s[1] + 1) - s[0]) * ((uniform_len - 1) / (s[1] - s[0]))
                    target_ticks = np.arange(uniform_len)
                    assert ticks[0] == target_ticks[0]
                    assert np.abs(ticks[-1]-target_ticks[-1]) < 1e-5, f"{ticks[-1]} vs {target_ticks[-1]}"
                    ticks[-1] = target_ticks[-1]

                    motions_resampled.append(
                        resample_motion(motion.iloc[s[0]: s[1] + 1, :], ticks, target_ticks, rotation_order).to_numpy())

                    scope_wav = [
                        s[0] * hop_len,
                        (s[1] + 1) * hop_len
                    ]
                    wavs_resampled.append(resample_wav_data(wav[scope_wav[0]: scope_wav[1]+1], uniform_len*hop_len))
                else:
                    ticks = (np.arange(s[0], s[1] + 1) - s[0]) * ((uniform_len) / (s[1] - s[0]))
                    target_ticks = np.arange(uniform_len + 1)
                    assert ticks[0] == target_ticks[0]
                    assert np.abs(ticks[-1]-target_ticks[-1]) < 1e-5, f"{ticks[-1]} vs {target_ticks[-1]}"
                    ticks[-1] = target_ticks[-1]

                    motions_resampled.append(
                        resample_motion(motion.iloc[s[0]: s[1] + 1, :], ticks, target_ticks, rotation_order).to_numpy()[
                        1:, :])

                    scope_wav = [
                        s[0] * hop_len,
                        (s[1] + 1) * hop_len
                    ]
                    wavs_resampled.append(resample_wav_data(wav[scope_wav[0]: scope_wav[1]+1], (uniform_len+1)*hop_len)[hop_len:])

            wav_resampled = np.concatenate(wavs_resampled)
            data_tmp = np.concatenate(motions_resampled)
            motion_resampled = pd.DataFrame(data=data_tmp,
                                            index=pd.to_timedelta(np.arange(data_tmp.shape[0]) / fps, unit='s'),
                                            columns=motion.columns)

            assert motion_resampled.shape[0]*hop_len == wav_resampled.shape[0]
            assert motion_resampled.shape[0] == uniform_len * len(scopes)

            mel_resampled = librosa.feature.melspectrogram(y=wav_resampled, sr=sr, n_fft=2048, hop_length=hop_len,
                                                           n_mels=mel.shape[1], fmin=0., fmax=8000)
            mel_resampled[np.where(mel_resampled == 0)] = 4e-6
            mel_resampled = np.transpose(np.log(mel_resampled))

            if mel_resampled.shape[0] > motion_resampled.shape[0]:
                mel_resampled = mel_resampled[: motion_resampled.shape[0]]
            elif mel_resampled.shape[0] < motion_resampled.shape[0]:
                motion_resampled = motion_resampled[: mel_resampled.shape[0]]

            assert mel_resampled.shape[0] == motion_resampled.shape[0], f"{mel_resampled.shape[0]} vs {motion_resampled.shape[0]}"

            if save:
                # wavfile.write(os.path.join(dir_save, n + f"{suffix}.wav"), sr, wav_resampled)
                np.save(os.path.join(dir_save, n + f"_scope{suffix}.npy"), np.array(scopes))
                np.save(os.path.join(dir_save, n + f"_index{suffix}.npy"), np.arange(counter, counter + len(scopes)))
                np.save(os.path.join(dir_save, n + f"_mel{suffix}.npy"), mel_resampled)
                motion_resampled.to_csv(os.path.join(dir_save, n + f"_motion{suffix}.csv"))

            res_scopes.append(np.array(scopes))
            res_mel.append(mel_resampled)
            res_motion.append(motion_resampled)

            print(n + f"_scope{suffix}:", res_scopes[-1].shape)
            print(n + f"_index{suffix}:", f"[{counter}, {counter + len(scopes) - 1}]")
            print(n + f"_mel{suffix}:", res_mel[-1].shape)
            print(n + f"_motion{suffix}:", res_motion[-1].shape)

            counter += len(scopes)

        return counter

    counter = process("", counter)

    return res_scopes, res_mel, res_motion