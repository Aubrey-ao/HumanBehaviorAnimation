# region Import.

import os
import sys
import pickle
import argparse
import json5

import numpy as np
import joblib as jl

from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

module_path = os.path.dirname(os.path.abspath(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)

from sampling import Inference

# endregion


__all__ = ["build_lexicon", "predict_lexeme"]


def get_args_parser():
    parser = argparse.ArgumentParser('gesture lexicon', add_help=False)

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--checkpoint_config', type=str, required=True)
    parser.add_argument('--lexicon_size', type=int, required=True)
    parser.add_argument('--num_kmeans_rerun', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save', action='store_true')

    return parser


"""
build lexicon using K-means.
1) mean and variance normalization for input features
2) shuffle input data order, re-run, pick the result according to inertia
reference: https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering 
"""
def build_lexicon(path_dataset: str, path_pretrained_net: str, path_config_train: str,
                  lexicon_size: int, num_rerun: int = 10, device: str = "cuda:0", save: bool = True):
    print('build lexicon...')

    data_dir = os.path.dirname(path_dataset)

    inference = Inference(path_dataset, path_pretrained_net, device, path_config_train)

    latent_code = inference.infer()  # num_clips X num_blocks X dim_feat.
    latent_code_reshaped = latent_code.reshape(-1, latent_code.shape[-1])  # (num_clips*num_blocks) X dim_feat.

    # normalization
    scaler = StandardScaler()
    latent_code_reshaped = scaler.fit_transform(latent_code_reshaped)
    jl.dump(scaler, os.path.join(data_dir, 'train_lexeme_scaler.sav'))

    # shuffle input order and re-run
    best_idx = None
    best_lxc = None
    best_inertia = None
    for idx in range(num_rerun):
        print('run', idx, 'in different data order...')
        x_shuffled = shuffle(latent_code_reshaped, random_state=idx)
        lxc = KMeans(n_clusters=lexicon_size, n_init=1, random_state=idx).fit(x_shuffled)

        if best_inertia is not None:
            if lxc.inertia_ < best_inertia:
                best_idx = idx
                best_lxc = lxc
                best_inertia = lxc.inertia_
        else:
            best_idx = idx
            best_lxc = lxc
            best_inertia = lxc.inertia_

    # final run
    print('final run in different data order...')
    x_shuffled = shuffle(latent_code_reshaped, random_state=best_idx)
    lexicon = KMeans(n_clusters=lexicon_size, init=best_lxc.cluster_centers_, n_init=1, random_state=num_rerun).fit(x_shuffled)

    labels_clip = lexicon.labels_.reshape(latent_code.shape[0], latent_code.shape[1]).astype(int)  # num_clips X num_blocks.
    lexemes = lexicon.cluster_centers_[labels_clip]  # num_clips X num_blocks X dim_feat.
    # lexemes_one_hot = np.eye(lexicon_size)[labels_clip]  # num_clips X num_blocks X dim_feat.

    print('lexeme:', lexemes.shape)
    print('lexeme_index:', labels_clip.shape)
    print('motion_latent_code:', latent_code.shape)

    if save:
        print('save...')
        data = dict(np.load(path_dataset))
        data["lexeme"] = lexemes
        data["lexeme_index"] = labels_clip
        data["motion_latent_code"] = latent_code
        np.savez(path_dataset, **data)

        with open(os.path.join(os.path.dirname(path_dataset), "lexicon.pkl"), "wb") as f:
            pickle.dump(lexicon, f)

    return lexicon, lexemes


def predict_lexeme(path_dataset: str, path_pretrained_net: str, path_config_train: str, path_lxm_scaler: str,
                   lexicon, device: str = "cuda:0", save: bool = True):
    print('predict lexemes...')

    inference = Inference(path_dataset, path_pretrained_net, device, path_config_train)

    latent_code = inference.infer()  # num_clips X num_blocks X dim_feat.
    latent_code_reshaped = latent_code.reshape(-1, latent_code.shape[-1])  # (num_clips*num_blocks) X dim_feat.

    # normalization
    scaler = jl.load(path_lxm_scaler)
    latent_code_reshaped = scaler.transform(latent_code_reshaped)

    labels = lexicon.predict(latent_code_reshaped)

    labels_clip = labels.reshape(latent_code.shape[0], latent_code.shape[1]).astype(int)  # num_clips X num_blocks.
    lexemes = lexicon.cluster_centers_[labels_clip, :]  # num_clips X num_blocks X dim_feat.
    # lexemes_one_hot = np.eye(lexicon_size)[labels_clip, :]  # num_clips X num_blocks X dim_feat.

    print('lexeme:', lexemes.shape)
    print('lexeme_index:', labels_clip.shape)
    print('motion_latent_code:', latent_code.shape)

    if save:
        print('save...')
        data = dict(np.load(path_dataset))
        data["lexeme"] = lexemes
        data["lexeme_index"] = labels_clip
        data["motion_latent_code"] = latent_code
        np.savez(path_dataset, **data)

    return lexemes


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    train_data_path = os.path.join(args.data_dir, 'train.npz')
    valid_data_path = os.path.join(args.data_dir, 'valid.npz')
    lxm_scaler_path = os.path.join(args.data_dir, 'train_lexeme_scaler.sav')

    with open(os.path.join(args.data_dir, 'config.json5'), 'r') as f:
        dataset_config = json5.load(f)
    dataset_config['lexicon_size'] = args.lexicon_size
    with open(os.path.join(args.data_dir, 'config.json5'), 'w') as f:
        json5.dump(dataset_config, f, indent=4)

    lexicon, _ = build_lexicon(train_data_path, args.checkpoint_path, args.checkpoint_config,
                               args.lexicon_size, args.num_kmeans_rerun, args.device, args.save)

    _ = predict_lexeme(valid_data_path, args.checkpoint_path, args.checkpoint_config, lxm_scaler_path,
                       lexicon, args.device, args.save)