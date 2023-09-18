__all__ = ["initialize_net", "align_motion_template_len_to_audio"]


def initialize_net(data_cfg, gen_cfg, lxm_intp_cfg):
    # init generator
    if gen_cfg['network']['name'] == "RNN":
        from Gesture_Generator.network import MotionGenerator_RNN
        gen = MotionGenerator_RNN(**gen_cfg['network']['hparams'])
    else:
        NotImplementedError

    # init lexeme interpreter
    from Lexeme_Interpreter.lxm_interpreter import LxmInterpreter
    lxm_intp = LxmInterpreter(data_cfg['lexicon_size'], data_cfg['num_blocks_per_clip']-1, data_cfg['uniform_len'],
                              **lxm_intp_cfg['network']['hparams'])

    return gen, lxm_intp


def align_motion_template_len_to_audio(aud_path, mo_path, rotation_order, save_path):
    import os
    import librosa
    import numpy as np
    from Data_Preprocessing.Utils.Pymo.BVH_loader import load as bvh_load
    from Data_Preprocessing.Utils.Pymo.BVH_loader import save as bvh_save
    
    audio, sr = librosa.load(aud_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    
    motion = bvh_load(mo_path)
    tgt_mo_len = int(duration * motion.fps)
    
    first_frame_rot = motion._joint_rotation[:1, ...]
    first_frame_trans = motion._joint_translation[:1, ...]
    motion._joint_rotation = np.repeat(first_frame_rot, tgt_mo_len, axis=0)
    motion._joint_translation =  np.repeat(first_frame_trans, tgt_mo_len, axis=0)
    motion._num_frames = tgt_mo_len
    motion._joint_orientation = None
    motion._joint_position = None
    motion.recompute_joint_global_info()
    
    bvh_save(motion, save_path, euler_order=rotation_order)