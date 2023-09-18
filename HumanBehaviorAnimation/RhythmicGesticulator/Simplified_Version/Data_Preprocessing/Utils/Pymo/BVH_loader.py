import numpy as np
from scipy.spatial.transform import Rotation as R 
from typing import List, Dict, Union

from .motion_data import MotionData

def load(fn:str, insert_T_pose:bool=False, ignore_root_offset=True, max_frames=None):
    with open(fn, 'r') as f:
        return load_from_io(f, fn, insert_T_pose, ignore_root_offset, max_frames)

def load_from_string(bvh_str:str, insert_T_pose:bool=False, ignore_root_offset=True, max_frames=None):
    import io
    return load_from_io(io.StringIO(bvh_str), 'str', insert_T_pose, ignore_root_offset, max_frames)

def load_from_io(f, fn='', insert_T_pose:bool=False, ignore_root_offset=True, max_frames=None):
    channels = []
    joints = []
    joint_parents = []
    joint_offsets = []
    end_sites = []
    fps = 0

    parent_stack = [None]
    for line in f:
        if 'ROOT' in line or 'JOINT' in line:
            joints.append(line.split()[-1])
            joint_parents.append(parent_stack[-1])
            channels.append(None)
            joint_offsets.append([0,0,0])

        elif 'End Site' in line:
            end_sites.append(len(joints))

            joints.append(parent_stack[-1] + '_end')
            joint_parents.append(parent_stack[-1])
            channels.append(None)
            joint_offsets.append([0,0,0])

        elif '{' in line:
            parent_stack.append(joints[-1])
        
        elif '}' in line:
            parent_stack.pop()

        elif 'OFFSET' in line:
            joint_offsets[-1] = [float(x) for x in line.split()[-3:]]

        elif 'CHANNELS' in line:
            trans_order = []
            trans_channels = []
            rot_order = []
            rot_channels = []
            for i, token in enumerate(line.split()):
                if 'position' in token:
                    trans_order.append(token[0])
                    trans_channels.append(i - 2)

                if 'rotation' in token:
                    rot_order.append(token[0])
                    rot_channels.append(i - 2)

            channels[-1] = [(''.join(trans_order), trans_channels),(''.join(rot_order), rot_channels)]

        elif 'Frame Time:' in line:
            _frame_time = float(line.split()[-1])
            print('frame time: ', _frame_time)
            fps = round(1. / _frame_time)
            break

    values = []
    for line in f:
        tokens = line.split()
        if len(tokens) == 0:
            break
        values.append([float(x) for x in tokens])
        if max_frames is not None and len(values) >= max_frames:
            break

    values = np.array(values)
    #values = values.reshape(values.shape[0],-1,3)
    if insert_T_pose:
        values = np.concatenate((np.zeros_like(values[:1]), values), axis=0)


    assert(parent_stack[0] is None)
    data = MotionData()
    data._fps = fps

    data._skeleton_joints = joints
    data._skeleton_joint_parents = [joints.index(n) if n is not None else -1 for n in joint_parents]
    data._skeleton_joint_offsets = np.array(joint_offsets)
    data._end_sites = end_sites

    if ignore_root_offset:
        data._skeleton_joint_offsets[0].fill(0)

    data._num_frames = values.shape[0]
    data._num_joints = len(data._skeleton_joints)
    
    data._joint_translation = np.zeros((data._num_frames, data._num_joints, 3))
    data._joint_rotation = np.zeros((data._num_frames, data._num_joints, 4))
    data._joint_rotation[:,:,-1] = 1

    value_idx = 0
    for i,ch in enumerate(channels):
        if ch == None:
            continue

        joint_num_channels = len(ch[0][1]) + len(ch[1][1])
        joint_values = values[:, value_idx:value_idx+joint_num_channels]
        value_idx += joint_num_channels

        if not ch[0][0] == '':
            data._joint_translation[:,i] = joint_values[:,ch[0][1]]
            if not ch[0] == 'XYZ':
                data._joint_translation[:,i] = data._joint_translation[:,i][:,[ord(c)-ord('X') for c in ch[0][0]]]

        if not ch[1][0] == '':
            rot = R.from_euler(ch[1][0], joint_values[:,ch[1][1]], degrees=True)
            data._joint_rotation[:,i] = rot.as_quat()

    print('loaded %d frames @ %d fps from %s' % (data._num_frames, data._fps, fn))

    data._joint_position = None
    data._joint_orientation = None
    data.align_joint_rotation_representation()
    data.recompute_joint_global_info()
    
    return data

def save(data:MotionData, fn:str, fmt:str='%10.6f', euler_order:str='XYZ', translational_joints=False, insert_T_pose:bool=False):
    with open(fn, 'w') as f:
        return save_to_io(data, f, fmt, euler_order, translational_joints, insert_T_pose)

def save_as_string(data:MotionData, fmt:str='%10.6f', euler_order:str='XYZ', translational_joints=False, insert_T_pose:bool=False):
    import io
    f = io.StringIO()
    save_to_io(data, f, fmt, euler_order, translational_joints, insert_T_pose)

    return f.getvalue()

def save_to_io(data:MotionData, f, fmt:str='%10.6f', euler_order:str='XYZ', translational_joints=False, insert_T_pose:bool=False):
    if not euler_order in ['XYZ', 'XZY', 'YZX', 'YXZ', 'ZYX', 'ZXY']:
        raise ValueError('euler_order ' + euler_order + ' is not supported!')

    # save header
    children = [[] for _ in range(data._num_joints)]
    for i,p in enumerate(data._skeleton_joint_parents[1:]):
        children[p].append(i+1)

    tab = ' '*4
    f.write('HIERARCHY\n')
    f.write('ROOT ' + data._skeleton_joints[0] + '\n')
    f.write('{\n')
    f.write(tab + 'OFFSET ' + ' '.join(fmt % s for s in data._skeleton_joint_offsets[0]) + '\n')
    f.write(tab + 'CHANNELS 6 Xposition Yposition Zposition ' + ' '.join(c+'rotation' for c in euler_order) + '\n')

    q = [(i, 1) for i in children[0][::-1]]
    last_level = 1
    output_order = [0]
    while len(q) > 0:
        idx, level = q.pop()
        output_order.append(idx)

        while last_level > level:
            f.write(tab * (last_level - 1) + '}\n')
            last_level -= 1

        indent = tab * level

        end_site = data._end_sites is not None and idx in data._end_sites
        if end_site:
            f.write(indent + 'End Site\n')
        else:
            f.write(indent + 'JOINT ' + data._skeleton_joints[idx] + '\n')

        f.write(indent + '{\n')
        level += 1
        indent += tab
        f.write(indent + 'OFFSET ' + ' '.join(fmt % s for s in data._skeleton_joint_offsets[idx]) + '\n')

        if not end_site:
            if translational_joints:
                f.write(indent + 'CHANNELS 6 Xposition Yposition Zposition ' + ' '.join(c+'rotation' for c in euler_order) + '\n')
            else:
                f.write(indent + 'CHANNELS 3 ' + ' '.join(c+'rotation' for c in euler_order) + '\n')

            q.extend([(i, level) for i in children[idx][::-1]])

        last_level = level

    while last_level > 0:
        f.write(tab * (last_level - 1) + '}\n')
        last_level -= 1

    f.write('MOTION\n')
    f.write('Frames: %d\n' % data.num_frames)
    f.write('Frame Time: ' + (fmt % (1/data.fps)) + '\n')

    # prepare channels
    value_idx = 0
    num_channels = 6 + (6 if translational_joints else 3) * (
        data._num_joints - 1 -
        (len(data._end_sites) if data._end_sites is not None else 0))
    values = np.zeros((data.num_frames, num_channels))
    for i in output_order:
        if data._end_sites is not None and i in data._end_sites:
            continue

        if i == 0 or translational_joints:
            values[:, value_idx:value_idx+3] = data._joint_translation[:,i]
            value_idx += 3

        rot = R.from_quat(data._joint_rotation[:,i])
        values[:, value_idx:value_idx+3] = rot.as_euler(euler_order, degrees=True)
        value_idx += 3

    # write frames
    if insert_T_pose:
        f.write(' '.join([fmt % 0] * num_channels))
        f.write('\n')
    f.write('\n'.join([' '.join(fmt % x for x in line) for line in values]))

