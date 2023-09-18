# region Import.

import os
import librosa

import numpy as np
import pyloudnorm as pyln

from typing import List

# endregion


__all__ = ["load_audio", "extract_melspec", "detect_onset", "prepare_audio_feature"]


def load_audio(dir_audio: str, names_file: List[str], dir_save: str,
               sr: int = 48000, normalize_loudness: bool = True, save: bool = False) -> List[np.ndarray]:
    res = []

    for n in names_file:
        path_audio = os.path.join(dir_audio, n + ".wav")
        audio, _ = librosa.load(path_audio, sr=sr)

        if normalize_loudness:
            meter = pyln.Meter(sr)  # create BS.1770 meter
            loudness = meter.integrated_loudness(audio)
            # loudness normalize audio to -20 dB LUFS
            audio = pyln.normalize.loudness(audio, loudness, -20.0)

        if save:
            np.save(os.path.join(dir_save, n + ".npy"), audio)

        res.append(audio)

        print(n + ':', res[-1].shape)

    return res


def extract_melspec(dir_loaded_audio: str, names_file: List[str], dir_save: str,
                    fps: int, sr: int, dim_mel: int, mel_filter_len: int, mel_hop_len: int,
                    save: bool = False) -> List[np.ndarray]:
    """
    Extract mel-spectrogram. 
    (Optional)Save generated mel-spectrogram(shape=(num_frames, n_mels)) to destination path.
    """

    res = []
    
    for name in names_file:
        audio = np.load(os.path.join(dir_loaded_audio, name+'.npy'))
        outfile = os.path.join(dir_save, name+'.npy')

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=mel_filter_len, hop_length=mel_hop_len, n_mels=dim_mel, fmin=20, fmax=7600)
        mel = librosa.power_to_db(mel, ref=np.max).T  # [L, D]
        mel = np.log(10 ** (mel/10))

        # interpolation
        from scipy.interpolate import griddata
        mo_len_pred = int(len(audio) / sr * fps)
        mel = griddata(
            np.arange(len(mel)),
            mel,
            sr / mel_hop_len / fps * np.arange(mo_len_pred),
            'linear'
        ).astype(np.float32)

        if save:
            np.save(outfile, mel)
        
        res.append(mel)
        
        print(name + ':', res[-1].shape)
    
    return res 


def detect_onset(dir_loaded_audio: str, names_file: List[str], dir_save: str,
                 fps: int, sr: int, bounds: List[float], save: bool = False) -> List[np.ndarray]:
    """
    Detect audio onset.
    (Optional)Save detected onset(shape=(num_onsets,), unit is `frame`) to destination path.
    """

    res = []
    
    for n in names_file:
        audio = np.load(os.path.join(dir_loaded_audio, n+".npy"))

        hop_length = int(sr / fps)
        
        spectral_novelty = librosa.onset.onset_strength(audio, sr=sr, aggregate=np.median,
                                                        fmin=20, fmax=7600, n_mels=256, hop_length=hop_length)
        
        wait = int(np.round(bounds[0] * fps))
        post_max = int(np.round(0.2 * fps))
        onset_times_spectral = librosa.onset.onset_detect(onset_envelope=spectral_novelty, sr=sr,
                                                          wait=wait,
                                                          pre_avg=1, post_avg=1, pre_max=1, post_max=post_max,
                                                          delta=0.0, hop_length=hop_length, units='time')
        onsets_spectral = librosa.time_to_frames(onset_times_spectral, sr=sr, hop_length=hop_length).astype(int)
        
        # region Filter onset.

        while True:
            onsets_selected = []
            empty_counter = 0
            for i, o in enumerate(onsets_spectral):
                if i == 0:
                    continue
                else:
                    if (o-onsets_spectral[i-1]) > int(bounds[1]*fps):
                        gap = int(bounds[1] * fps / 2)
                        s = int(onsets_spectral[i-1] + gap)
                        e = int(o - gap)
                        assert e >= s

                        spectral_novelty_tmp = spectral_novelty[s: e+1]
                        max_tmp = np.max(spectral_novelty_tmp)
                        if max_tmp > 0.01:
                            onset_selected = np.argmax(spectral_novelty_tmp) + s
                            onsets_selected.append(onset_selected)
                        else:
                            duration = o - onsets_spectral[i-1]
                            N = int(np.median(np.arange(np.ceil(duration/int(bounds[1]*fps)),
                                                        np.floor(duration/gap)+1).astype(int)))
                            step = int(duration / N)

                            onset_selected_tmp = np.arange(onsets_spectral[i-1], o, step)
                            assert (len(onset_selected_tmp) > 1) and (onset_selected_tmp[0] == onsets_spectral[i-1])

                            onset_selected = list(onset_selected_tmp[1:][: (N-1)])
                            onsets_selected += onset_selected

                            # print(duration/fps, duration, N, step, [onset_spectral[i-1], o], onset_selected)

                            empty_counter += 1

            # print(empty_counter)
            # print(len(onsets_selected))
            onset_spectral_new = np.sort(np.concatenate([onsets_spectral, np.array(onsets_selected)]))
            onset_times_spectral_new = librosa.frames_to_time(onset_spectral_new, sr=sr, hop_length=hop_length)
            onsets_spectral = onset_spectral_new
            onset_times_spectral = onset_times_spectral_new

            if np.max(np.abs(np.diff(np.array(onset_times_spectral)))) <= (bounds[1]+(1/fps)):
                break

        assert np.min(np.abs(np.diff(np.array(onset_times_spectral)))) >= (bounds[0]-(1/fps))
        
        # endregion
        
        mean_dur_onset = np.mean(np.abs(np.diff(onset_times_spectral)))
        median_dur_onset = np.median(np.abs(np.diff(onset_times_spectral)))
        min_dur_onset = np.min(np.abs(np.diff(onset_times_spectral)))
        max_dur_onset = np.max(np.abs(np.diff(onset_times_spectral)))
        
        if save:
            np.save(os.path.join(dir_save, n + ".npy"), onsets_spectral)
        
        res.append(onsets_spectral)
        
        print(n + ':', res[-1].shape,
              f"Mean interval: {mean_dur_onset}",
              f"Median interval: {median_dur_onset}",
              f"Max interval: {max_dur_onset}",
              f"Min interval: {min_dur_onset}",
              f"Wait: {wait}", f"Post_max: {post_max}",
              f"Filtered: {len(onsets_selected)}", f"Unfiltered: {empty_counter}")
    
    return res


def prepare_audio_feature(dir_data_len_uniform: str, names_file: List[str], dir_save: str,
                          uniform_len: int, dim_pos_enc: int, use_pos_enc: bool = False,
                          save: bool = False) -> List[np.ndarray]:
    if use_pos_enc:
        pos_enc = np.zeros((uniform_len, dim_pos_enc))
        position = np.arange(uniform_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim_pos_enc, 2) * -(np.log(10000.0) / dim_pos_enc))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        pos_enc = pos_enc[-1::-1, :]
    
    res = []
    
    def process(suffix: str = ""):
        for n in names_file:
            mel = np.load(os.path.join(dir_data_len_uniform, n + f"_mel{suffix}.npy"))
            
            assert mel.shape[0] % uniform_len == 0

            if use_pos_enc:
                times = int(mel.shape[0] / uniform_len)
                pos_encs = np.tile(pos_enc, (times, 1))
                new_mel = np.concatenate([mel, pos_encs], axis=-1)
            else:
                new_mel = mel
            
            if save:
                np.save(os.path.join(dir_save, n + f"{suffix}.npy"), new_mel)
            
            res.append(new_mel)
            
            print(n + f"{suffix}:", new_mel.shape)
    
    process()
    
    return res