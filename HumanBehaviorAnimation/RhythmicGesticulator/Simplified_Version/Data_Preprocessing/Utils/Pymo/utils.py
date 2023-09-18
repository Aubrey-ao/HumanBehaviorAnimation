import numpy as np

from scipy.spatial.transform import Rotation as R 

def quat_product(p:np.ndarray, q:np.ndarray):
    if p.shape[-1] != 4 or q.shape[-1] != 4:
        raise ValueError('operands should be quaternions')

    if len(p.shape) != len(q.shape):
        if len(p.shape) == 1:
            p.reshape([1]*(len(q.shape)-1)+[4])
        elif len(q.shape) == 1:
            q.reshape([1]*(len(p.shape)-1)+[4])
        else:
            raise ValueError('mismatching dimensions')

    is_flat = len(p.shape) == 1
    if is_flat:
        p = p.reshape(1,4)
        q = q.reshape(1,4)
    
    product = np.empty([ max(p.shape[i], q.shape[i]) for i in range(len(p.shape)-1) ] + [4], dtype=np.result_type(p.dtype, q.dtype))
    product[..., 3] = p[..., 3] * q[..., 3] - np.sum(p[..., :3] * q[..., :3], axis=-1)
    product[..., :3] = (p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] +
                      np.cross(p[..., :3], q[..., :3]))

    if is_flat:
        product = product.reshape(4)

    return product

def flip_vector(vt:np.ndarray, normal:np.ndarray, inplace:bool):
    vt = np.asarray(vt).reshape(-1,3)
    normal = np.asarray(normal).reshape(-1,3)
    if inplace:
        vt -= (2 * np.sum(vt*normal, axis=-1, keepdims=True)) * normal
        return vt
    else:
        return vt - (2 * np.sum(vt*normal, axis=-1, keepdims=True)) * normal

def flip_quaternion(qt:np.ndarray, normal:np.ndarray, inplace:bool):
    qt = np.asarray(qt).reshape(-1,4)
    normal = np.asarray(normal).reshape(-1,3)

    if not inplace:
        qt = qt.copy()
    flip_vector(qt[:,:3], normal, True)
    qt[:,-1] = -qt[:,-1]
    return qt

def align_angles(a:np.ndarray, degrees:bool, inplace:bool):
    ''' make the angles in the array continuous

        we assume the first dim of a is the time
    '''
    step = 360. if degrees else np.pi*2

    a = np.asarray(a)
    diff = np.diff(a, axis=0)
    num_steps = np.round(diff / step)
    num_steps = np.cumsum(num_steps, axis=0)
    if not inplace:
        a = a.copy()
    a[1:] -= num_steps * step

    return a

def align_quaternion(qt:np.ndarray, inplace:bool):
    ''' make q_n and q_n+1 in the same semisphere

        the first axis of qt should be the time
    '''
    qt = np.asarray(qt)
    if qt.shape[-1] != 4:
        raise ValueError('qt has to be an array of quaterions')
    
    if not inplace:
        qt = qt.copy()

    if qt.size == 4: # do nothing since there is only one quation
        return qt

    sign = np.sum(qt[:-1]*qt[1:], axis=-1)
    sign[sign < 0] = -1
    sign[sign >= 0] = 1
    sign = np.cumprod(sign, axis=0,)

    qt[1:][sign < 0] *= -1

    return qt

def extract_heading_Y_up(q:np.ndarray):
    ''' extract the rotation around Y axis from given quaternions
        
        note the quaterions should be {(x,y,z,w)}
    '''
    q = np.asarray(q)
    shape = q.shape
    q = q.reshape(-1, 4)

    v = R(q,True,False).as_dcm()[:,:,1]

    #axis=np.cross(v,(0,1,0))
    axis = v[:,(2,1,0)]
    axis *= [-1,0,1]
    
    norms = np.linalg.norm(axis,axis=-1)
    scales = np.empty_like(norms)
    small_angle = (norms <= 1e-3)
    large_angle = ~small_angle

    scales[small_angle] = norms[small_angle] + norms[small_angle]**3 / 6
    scales[large_angle] = np.arccos(v[large_angle,1]) / norms[large_angle]

    correct = R.from_rotvec(axis*scales[:,None])

    heading = (correct*R(q,True,False)).as_quat()
    heading[heading[:,-1] < 0] *= -1

    return heading.reshape(shape)

def extract_heading_frame_Y_up(root_pos, root_rots):
    heading = extract_heading_Y_up(root_rots)

    pos = np.copy(root_pos)
    pos[...,1] = 0

    return pos,heading
    
def get_joint_color(names, left='r', right='b', otherwise='y'):
    matches = (
        ('l', 'r'),
        ('L', 'R'),
        ('left', 'right'),
        ('Left', 'Right'),
        ('LEFT', 'RIGHT')
    )

    def check(n, i):
        for m in matches:
            if n[:len(m[i])] == m[i] and m[1-i] + n[len(m[i]):] in names:
                return True
                
            if n[-len(m[i]):] == m[i] and n[:-len(m[i])] + m[1-i] in names:
                return True

        return False

    color = [left if check(n, 0) else right if check(n, 1) else otherwise for n in names]
    return color

def animate_motion_data(data, show_skeleton=True, show_animation=True, interval=1):
    if (not show_skeleton) and (not show_animation):
        return

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D


    parent_idx = np.array(data._skeleton_joint_parents)
    parent_idx[0] = 0

    joint_colors=get_joint_color(data.joint_names)
    
    if data.end_sites is not None:
        for i in range(len(joint_colors)):
            if i in data.end_sites:
                joint_colors[i] = 'k'

    #############################
    # draw skeleton
    if show_skeleton:
        ref_joint_positions = data.get_reference_pose()
        tmp = ref_joint_positions.reshape(-1,3)
        bound = np.array([np.min(tmp, axis=0), np.max(tmp, axis=0)])
        bound[1,:] -= bound[0,:]
        bound[1,:] = np.max(bound[1,:])
        bound[1,:] += bound[0,:]

        fig = plt.figure(figsize=(10,10)) 
        ax = fig.add_subplot('111', projection='3d')
        
        #ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        
        pos = ref_joint_positions
        strokes = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]

        ax.auto_scale_xyz(bound[:,0], -bound[:,2], bound[:,1])

    ########################################
    # animate motion
    if show_animation:
        joint_pos = data._joint_position
        tmp = joint_pos[:1].reshape(-1,3)
        bound = np.array([np.min(tmp, axis=0), np.max(tmp, axis=0)])
        bound[1,:] -= bound[0,:]
        bound[1,:] = np.max(bound[1,:])
        bound[1,:] += bound[0,:]

        fig = plt.figure(figsize=(10,10)) 
        ax = fig.add_subplot('111', projection='3d')
        
        #ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        
        pos = joint_pos[0]
        strokes = [plt.plot(xs=pos[(i,p),0], zs=pos[(i,p),1], ys=-pos[(i,p),2], c=joint_colors[i], marker='x', linestyle='solid') for (i,p) in enumerate(parent_idx)]

        ax.auto_scale_xyz(bound[:,0], -bound[:,2], bound[:,1])

        def update_lines(num):
            for (i,p) in enumerate(parent_idx):
                strokes[i][0].set_data(joint_pos[num][(i,p),0], -joint_pos[num][(i,p),2])
                strokes[i][0].set_3d_properties(joint_pos[num][(i,p),1])
            plt.title('frame {num}'.format(num=num))

        line_ani = animation.FuncAnimation(
            fig, update_lines, joint_pos.shape[0],
            interval=interval, blit=False)

    plt.show()