# region Import.

import joblib as jl

from typing import List
from sklearn.pipeline import Pipeline

from .parsers import BVHParser
from .preprocessing import *
from .utils import *
from .Pymo.BVH_loader import load as bvh_load
from .Pymo.BVH_loader import save as bvh_save

# endregion


__all__ = ["extract_bvh_motion", "transfer_to_facing_coordinate", "make_motion_template", "prepare_motion_feature"]


def extract_bvh_motion(dir_motion: str, names_file: List[str], dir_save: str,
                       fps: int, joints_selected: List[str], save: bool = False) -> List[pd.DataFrame]:
    parser = BVHParser()
    
    num_files = len(names_file)
    
    datas = []
    for name in names_file:
        datas.append(parser.parse(os.path.join(dir_motion, name + '.bvh')))
    
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        ('jtsel', JointSelector(joints_selected, include_root=True)),
        ('cnst', ConstantsRemover())
    ])
    
    data_outs = data_pipe.fit_transform(datas)
    res = [data.values for data in data_outs]
    
    jl.dump(data_pipe, os.path.join(dir_save, 'data_pipe.sav'))
    
    if save:
        for i in range(num_files):
            res[i].to_csv(os.path.join(dir_save, names_file[i] + ".csv"))
    
    print(" ")
    for i in range(num_files):
        print(names_file[i] + ':', res[i].shape)
    
    return res


def transfer_to_facing_coordinate(dir_motion: str, names_file: List[str], dir_save: str,
                                  rotation_order: str):
    for name in names_file:
        motion_data = bvh_load(os.path.join(dir_motion, name + ".bvh"))
        motion_data = motion_data.to_facing_coordinate()

        bvh_save(motion_data, os.path.join(dir_save, name + ".bvh"), euler_order=rotation_order)


def make_motion_template(dir_motion: str, names_file: List[str], dir_save: str, 
                         fps: int, rotation_order: str, motion_duration: int = 300):
    for name in names_file:
        motion_data = bvh_load(os.path.join(dir_motion, name + ".bvh"))
        motion_data.resample(fps)
        tgt_mo_len = int(fps * motion_duration)
        
        first_frame_rot = motion_data._joint_rotation[:1, ...]
        first_frame_trans = motion_data._joint_translation[:1, ...]
        motion_data._joint_rotation = np.repeat(first_frame_rot, tgt_mo_len, axis=0)
        motion_data._joint_translation =  np.repeat(first_frame_trans, tgt_mo_len, axis=0)
        motion_data._num_frames = tgt_mo_len
        motion_data._joint_orientation = None
        motion_data._joint_position = None
        motion_data.recompute_joint_global_info()
        
        bvh_save(motion_data, os.path.join(dir_save, "motion_template.bvh"), euler_order=rotation_order)
        break
        


def prepare_motion_feature(dir_data_len_uniform: str, names_file: List[str], dir_save: str,
                           root_joint: str, rotation_order: str, save: bool = False) -> List[pd.DataFrame]:
    res = []

    def process(suffix: str = ""):
        for n in names_file:
            motion = pd.read_csv(os.path.join(dir_data_len_uniform, n + f"_motion{suffix}.csv"), index_col=0)
            
            motion_filtered = motion.drop([c for c in motion.columns if (('position' in c) and (root_joint+'_' not in c))], axis=1)
            
            new_motion = euler_to_expmap(motion_filtered, rotation_order)
            
            if save:
                new_motion.to_csv(os.path.join(dir_save, n + f"{suffix}.csv"))
            
            res.append(new_motion)
            
            print(n + f"{suffix}:", res[-1].shape)
    
    process()
    
    return res