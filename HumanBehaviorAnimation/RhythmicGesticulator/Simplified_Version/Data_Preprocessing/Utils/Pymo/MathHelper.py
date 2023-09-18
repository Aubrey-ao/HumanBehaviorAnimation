from enum import IntEnum
import numpy as np
import os
from typing import Tuple, List, Union, Iterable, Optional
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
# Note: scipy==1.6.x is compiled with cython, which is not suitable for debugging in pycharm.
# Zhenhua Song: if we import torch or some large library here, there will not be enough memory for run mpi samcon..


fdir = os.path.dirname(__file__)


class RotateType(IntEnum):
    Matrix = 1
    AxisAngle = 2
    Vec6d = 3
    SVD9d = 4
    Quaternion = 5


class MathHelper:

    @staticmethod
    def quat_from_other_rotate(x: np.ndarray, rotate_type: RotateType) -> np.ndarray:
        if rotate_type == RotateType.Matrix or rotate_type == RotateType.SVD9d:
            return MathHelper.matrix_to_quat(x)
        elif rotate_type == RotateType.Quaternion:
            return x
        elif rotate_type == RotateType.AxisAngle:
            return MathHelper.quat_from_axis_angle(x)
        elif rotate_type == RotateType.Vec6d:
            return MathHelper.vec6d_to_quat(x)
        else:
            raise NotImplementedError

    @staticmethod
    def quat_to_other_rotate(quat: np.ndarray, rotate_type: RotateType):
        if rotate_type == RotateType.SVD9d or rotate_type == RotateType.Matrix:
            return MathHelper.quat_to_matrix(quat)
        elif rotate_type == RotateType.Vec6d:
            return MathHelper.quat_to_vec6d(quat)
        elif rotate_type == RotateType.Quaternion:
            return quat
        elif rotate_type == RotateType.AxisAngle:
            return MathHelper.quat_to_axis_angle(quat)
        else:
            raise NotImplementedError

    @staticmethod
    def get_rotation_dim(rotate_type: RotateType):
        if rotate_type == RotateType.Vec6d:
            return 6
        elif rotate_type == RotateType.AxisAngle:
            return 3
        elif rotate_type == RotateType.SVD9d:
            return 9
        elif rotate_type == RotateType.Matrix:
            return 9
        elif rotate_type == RotateType.Quaternion:
            return 4
        else:
            raise NotImplementedError

    @staticmethod
    def get_rotation_last_shape(rotate_type: RotateType) -> Tuple:
        if rotate_type == RotateType.Vec6d:
            last_shape = (3, 2)
        elif rotate_type == RotateType.AxisAngle:
            last_shape = (3,)
        elif rotate_type == RotateType.SVD9d:
            last_shape = (3, 3)
        elif rotate_type == RotateType.Matrix:
            last_shape = (3, 3)
        elif rotate_type == RotateType.Quaternion:
            last_shape = (4,)
        else:
            raise NotImplementedError

        return last_shape

    @staticmethod
    def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        multiply 2 quaternions. p.shape == q.shape
        """
        assert 2 == len(p.shape) and 4 == p.shape[-1]
        assert 2 == len(q.shape) and 4 == q.shape[-1]

        w: np.ndarray = p[:, 3:4] * q[:, 3:4] - np.sum(p[:, :3] * q[:, :3], axis=1, keepdims=True)
        xyz: np.ndarray = p[:, None, 3] * q[:, :3] + q[:, None, 3] * p[:, :3] + np.cross(p[:, :3], q[:, :3], axis=1)

        return np.concatenate([xyz, w], axis=-1)

    @staticmethod
    def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float):
        """
        update quaternion, q_{t+1} = normalize(q_{t} + 0.5 * w * q_{t})
        """
        init_q_shape = q.shape
        if q.shape[-1] == 1 and omega.shape[-1] == 1:
            q = q.reshape(q.shape[:-1])
            omega = omega.reshape(omega.shape[:-1])
        assert 4 == q.shape[-1] and 3 == omega.shape[-1]

        omega = np.concatenate([omega, np.zeros(omega.shape[:-1] + (1,))], axis=-1)

        delta_q = 0.5 * dt * MathHelper.quat_multiply(omega, q)
        result = q + delta_q
        result /= np.linalg.norm(result, axis=-1, keepdims=True)
        return result.reshape(init_q_shape)

    @staticmethod
    def quat_to_matrix(q: np.ndarray) -> np.ndarray:
        assert q.shape[-1] == 4
        return Rotation(q.reshape(-1, 4), copy=False, normalize=False).as_matrix().reshape(q.shape[:-1] + (3, 3))

    @staticmethod
    def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
        assert mat.shape[-2:] == (3, 3)
        return Rotation.from_matrix(mat.reshape((-1, 3, 3))).as_quat().reshape(mat.shape[:-2] + (4,))

    @staticmethod
    def quat_to_vec6d(q: np.ndarray) -> np.ndarray:
        """
        input quaternion in shape (..., 4)
        return: in shape (..., 3, 2)
        """
        assert q.shape[-1] == 4
        mat: np.ndarray = Rotation(q.reshape(-1, 4)).as_matrix()  # shape == (N, 3, 3)
        mat: np.ndarray = mat.reshape(q.shape[:-1] + (3, 3))
        return mat[..., :2]

    @staticmethod
    def vec6d_to_quat(x: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        input 6d vector in shape (..., 3, 2)
        return in shape (..., 4)
        """
        assert x.shape[-2:] == (3, 2)
        if normalize:
            x = x / np.linalg.norm(x, axis=-2, keepdims=True)

        last_col: np.ndarray = np.cross(x[..., 0], x[..., 1], axis=-1)
        last_col = last_col / np.linalg.norm(last_col, axis=-1, keepdims=True)

        mat = np.concatenate([x, last_col[..., None]], axis=-1)
        quat: np.ndarray = Rotation.from_matrix(mat.reshape((-1, 3, 3))).as_quat().reshape(x.shape[:-2] + (4,))
        return quat

    @staticmethod
    def normalize_angle(a: np.ndarray) -> np.ndarray:
        """
        Covert angles to [-pi, pi)
        """
        res: np.ndarray = a.copy()
        res[res >= np.pi] -= 2 * np.pi
        res[res < np.pi] += 2 * np.pi
        return res

    @staticmethod
    def normalize_vec(a: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(a, axis=-1, keepdims=True)
        norm[norm == 0] = 1
        return a / norm

    @staticmethod
    def up_vector() -> np.ndarray:
        """
        return (0, 1, 0)
        """
        return np.array([0.0, 1.0, 0.0])

    @staticmethod
    def ego_forward_vector() -> np.ndarray:
        """
        return (0, 0, 1)
        """
        return np.array([0.0, 0.0, 1.0])

    @staticmethod
    def unit_vector(axis: int) -> np.ndarray:  # shape == (3,)
        res = np.zeros(3)
        res[axis] = 1
        return res

    @staticmethod
    def unit_quat_scipy() -> np.ndarray:  # shape == (4,)
        return MathHelper.unit_quat()

    @staticmethod
    def unit_quat_scipy_list() -> List[float]:
        return [0.0, 0.0, 0.0, 1.0]

    @staticmethod
    def quat_from_scipy_to_ode(q: np.ndarray) -> np.ndarray:
        return MathHelper.xyzw_to_wxyz(q)

    @staticmethod
    def quat_from_ode_to_scipy(q: np.ndarray) -> np.ndarray:
        return MathHelper.wxyz_to_xyzw(q)

    @staticmethod
    def quat_from_ode_to_unity(q: np.ndarray) -> np.ndarray:
        return MathHelper.wxyz_to_xyzw(q)

    @staticmethod
    def unit_quat_ode() -> np.ndarray:
        return np.array(MathHelper.unit_quat_ode_list())

    @staticmethod
    def unit_quat_ode_list() -> List[float]:
        return [1.0, 0.0, 0.0, 0.0]

    @staticmethod
    def unit_quat_unity() -> np.ndarray:
        return np.asarray(MathHelper.unit_quat_unity_list())

    @staticmethod
    def unit_quat_unity_list() -> List[float]:
        return [0.0, 0.0, 0.0, 1.0]

    @staticmethod
    def unit_quat() -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0])

    @staticmethod
    def unit_quat_arr(shape: Union[int, Iterable, Tuple[int]]) -> np.ndarray:
        if type(shape) == int:
            shape = (shape, 4)

        res = np.zeros(shape)
        res[..., -1] = 1
        return res.reshape(shape)

    @staticmethod
    def ode_quat_to_rot_mat(q: np.ndarray) -> np.ndarray:
        return Rotation(MathHelper.quat_from_ode_to_scipy(q)).as_matrix()

    @staticmethod
    def rot_mat_to_ode_quat(mat: np.ndarray) -> np.ndarray:
        return MathHelper.quat_from_scipy_to_ode(Rotation.from_matrix(mat).as_quat())

    @staticmethod
    def vec_diff(v_in: np.ndarray, forward: bool, fps: float):
        v = np.empty_like(v_in)
        frag = v[:-1] if forward else v[1:]
        frag[:] = np.diff(v_in, axis=0) * fps
        v[-1 if forward else 0] = v[-2 if forward else 1]
        return v

    @staticmethod
    def vec_axis_to_zero(v: np.ndarray, axis: Union[int, List[int], np.ndarray]) -> np.ndarray:
        res: np.ndarray = v.copy()
        res[..., axis] = 0
        return res

    @staticmethod
    def xz_vector_to_xyz(xz: np.ndarray) -> np.ndarray:
        xyz = np.zeros((xz.shape[:-1] + (3,)))
        xyz[..., [0, 2]] = xz
        return xyz

    @staticmethod
    def flip_quat_by_w(q: np.ndarray) -> np.ndarray:
        res = q.copy()
        idx: np.ndarray = res[..., -1] < 0
        res[idx, :] = -res[idx, :]
        return res

    @staticmethod
    def flip_quat_arr_by_w(*args):
        return [MathHelper.flip_quat_by_w(i) for i in args]

    @staticmethod
    def flip_vector_by_dot(x: np.ndarray, inplace: bool = False) -> np.ndarray:
        """
        make sure x[i] * x[i+1] >= 0
        """
        if x.ndim == 1:
            return x

        sign: np.ndarray = np.sum(x[:-1] * x[1:], axis=-1)
        sign[sign < 0] = -1
        sign[sign >= 0] = 1
        sign = np.cumprod(sign, axis=0, )

        x_res = x.copy() if not inplace else x
        x_res[1:][sign < 0] *= -1

        return x_res

    @staticmethod
    def flip_vec3_by_dot(x: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert x.shape[-1] == 3
        return MathHelper.flip_vector_by_dot(x, inplace)

    @staticmethod
    def flip_quat_by_dot(q: np.ndarray, inplace: bool = False) -> np.ndarray:
        if q.shape[-1] != 4:
            raise ValueError

        return MathHelper.flip_vector_by_dot(q, inplace)

    @staticmethod
    def flip_quat_arr_by_dot(*args) -> List[np.ndarray]:
        return [MathHelper.flip_quat_by_dot(i) for i in args]

    @staticmethod
    def flip_quat_pair_by_dot(q0s: np.ndarray, q1s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        q0 will not be changed.
        q1 will be flipped to the same semi sphere as q0
        """
        assert q0s.shape[-1] == 4
        assert q1s.shape == q0s.shape

        dot_value = np.sum(q0s * q1s, axis=-1, keepdims=True) < 0
        dot_res = np.concatenate([dot_value] * 4, axis=-1)
        q1: np.ndarray = q1s.copy()
        q1[dot_res] = -q1[dot_res]
        return q0s, q1

    @staticmethod
    def quat_equal(q1: np.ndarray, q2: np.ndarray) -> bool:
        return np.all(np.abs(MathHelper.flip_quat_by_w(q1) - MathHelper.flip_quat_by_w(q2)) < 1e-5)

    @staticmethod
    def proj_vec_to_plane(a: np.ndarray, v: np.ndarray):
        """
        Project Vector to Plane
        :param a: original vector
        :param v: Normal vector of Plane
        :return: a_new(result of projection)
        """
        # k: coef.
        # a_new = a - k * v
        # a_new * v = 0
        # Solution: k = (a * v) / (v * v)
        k: np.ndarray = np.sum(a * v, axis=-1) / np.sum(v * v, axis=-1)  # (N, )
        return a - np.repeat(k, 3).reshape(v.shape) * v

    @staticmethod
    def proj_multi_vec_to_a_plane(a_arr: np.ndarray, v: np.ndarray):
        v_arr = np.zeros_like(a_arr)
        v_arr[:, :] = v
        return MathHelper.proj_vec_to_plane(a_arr, v_arr)

    @staticmethod
    def quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Rotation from vector a to vector b
        :param a: (n, 3) vector
        :param b: (n, 3) vector
        :return: (n, 4) quaternion
        """
        cross_res = np.cross(a, b)
        w_ = np.sqrt((a ** 2).sum(axis=-1) * (b ** 2).sum(axis=-1)) + (a * b).sum(axis=-1)
        res_ = np.concatenate([cross_res, w_[..., np.newaxis]], axis=-1)
        return res_ / np.linalg.norm(res_, axis=-1, keepdims=True)

    @staticmethod
    def quat_to_axis_angle(q: np.ndarray, normalize=True, copy=True):
        assert q.shape[-1] == 4
        return Rotation(q.reshape((-1, 4)), normalize=normalize, copy=copy).as_rotvec().reshape(q.shape[:-1] + (3, ))

    @staticmethod
    def quat_from_axis_angle(axis: np.ndarray, angle: Optional[np.ndarray] = None, normalize: bool = False) -> np.ndarray:
        if angle is not None:
            assert axis.shape == angle.shape + (3,)
            if normalize:
                axis: np.ndarray = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
            angle: np.ndarray = (0.5 * angle)[..., None]
            sin_res: np.ndarray = np.sin(angle)
            cos_res: np.ndarray = np.cos(angle)
            result = np.concatenate([axis * sin_res, cos_res], axis=-1)
            return result
        else:
            return Rotation.from_rotvec(axis.reshape((-1, 3))).as_quat().reshape(axis.shape[:-1] + (4,))

    @staticmethod
    def log_quat(q: np.ndarray) -> np.ndarray:
        """
        log quaternion。
        :param q: (n, 4) quaternion
        :return:
        """
        if q.shape[-1] != 4:
            raise ArithmeticError
        if q.ndim > 1:
            return 0.5 * Rotation(q.reshape(-1, 4), copy=False).as_rotvec().reshape(q.shape[:-1] + (3,))
        else:
            return 0.5 * Rotation(q, copy=False).as_rotvec()

    @staticmethod
    def exp_to_quat(v: np.ndarray) -> np.ndarray:
        """

        :param v:
        :return:
        """
        # Note that q and -q is the same rotation. so result is not unique
        if v.shape[-1] != 3:
            raise ArithmeticError
        if v.ndim > 1:
            return Rotation.from_rotvec(2 * v.reshape(-1, 3)).as_quat().reshape(v.shape[:-1] + (4,))
        else:
            return Rotation.from_rotvec(2 * v).as_quat()

    @staticmethod
    def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
        return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)

    @staticmethod
    def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
        return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)

    @staticmethod
    def facing_decompose(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        return: ry, facing. ry only has y component, and facing only has (x, z) component.
        """
        return MathHelper.y_decompose(q)

    @staticmethod
    def extract_heading_Y_up(q: np.ndarray):
        """ extract the rotation around Y axis from given quaternions

            note the quaterions should be {(x,y,z,w)}
        """
        q = np.asarray(q)
        shape = q.shape
        q = q.reshape(-1, 4)

        v = Rotation(q, normalize=False, copy=False).as_matrix()[:, :, 1]

        # axis=np.cross(v,(0,1,0))
        axis = v[:, (2, 1, 0)]
        axis *= [-1, 0, 1]

        norms = np.linalg.norm(axis, axis=-1)
        scales = np.empty_like(norms)
        small_angle = (norms <= 1e-3)
        large_angle = ~small_angle

        scales[small_angle] = norms[small_angle] + norms[small_angle] ** 3 / 6
        scales[large_angle] = np.arccos(v[large_angle, 1]) / norms[large_angle]

        correct = Rotation.from_rotvec(axis * scales[:, None])

        heading = (correct * Rotation(q, normalize=False, copy=False)).as_quat()
        heading[heading[:, -1] < 0] *= -1

        return heading.reshape(shape)

    @staticmethod
    def decompose_rotation(q: np.ndarray, vb: np.ndarray):
        rot_q = Rotation(q, copy=False)
        va = rot_q.apply(vb)
        va /= np.linalg.norm(va, axis=-1, keepdims=True)

        rot_axis = np.cross(va, vb)
        rot_axis_norm = np.linalg.norm(rot_axis, axis=-1, keepdims=True)
        rot_axis_norm[rot_axis_norm < 1e-14] = 1e-14
        rot_axis /= rot_axis_norm

        rot_angle = np.asarray(-np.arccos(np.clip(va.dot(vb), -1, 1))).reshape(-1)
        # TODO: minus or plus..?

        if rot_axis.ndim > 1:
            rot_angle = rot_angle.reshape(-1, 1)

        ret_result: np.ndarray = (Rotation.from_rotvec(rot_angle * (-rot_axis)) * rot_q).as_quat()
        ret_result: np.ndarray = MathHelper.flip_quat_by_dot(ret_result)
        return ret_result

    @staticmethod
    def axis_decompose(q: np.ndarray, axis: np.ndarray):
        """
        return:
        res: rotation along axis
        r_other:
        """
        assert axis.ndim == 1 and axis.shape[0] == 3
        res = MathHelper.decompose_rotation(q, np.asarray(axis))
        r_other = (Rotation(res, copy=False, normalize=False).inv() * Rotation(q, copy=False, normalize=False)).as_quat()
        r_other = MathHelper.flip_quat_by_dot(r_other)
        res[np.abs(res) < 1e-14] = 0
        r_other[np.abs(r_other) < 1e-14] = 0
        res /= np.linalg.norm(res, axis=-1, keepdims=True)
        r_other /= np.linalg.norm(r_other, axis=-1, keepdims=True)
        return res, r_other

    @staticmethod
    def x_decompose(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return MathHelper.axis_decompose(q, np.array([1.0, 0.0, 0.0]))

    @staticmethod
    def y_decompose(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return MathHelper.axis_decompose(q, np.array([0.0, 1.0, 0.0]))

    @staticmethod
    def z_decompose(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return MathHelper.axis_decompose(q, np.array([0.0, 0.0, 1.0]))

    # @staticmethod
    # def slerp_joint_rotation(q: np.ndarray, new_frame: int):
    #    frame, joint, _ = q.shape

    @staticmethod
    def resample_joint_linear(x: np.ndarray, ratio: int, old_fps: int):
        old_frame: int = x.shape[0]
        new_frame: int = old_frame * ratio
        new_fps: int = old_fps * ratio
        result: np.ndarray = np.zeros((new_frame,) + x.shape[1:])
        nj: int = x.shape[1]
        ticks = np.arange(0, old_frame, dtype=np.float64) / old_fps
        new_ticks = np.arange(0, new_frame, dtype=np.float64) / new_fps
        for index in range(nj):
            j_interp = interp1d(ticks, x[:, index], kind='linear', axis=0, copy=True, bounds_error=False, assume_sorted=True)
            result[:, index] = j_interp(new_ticks)
        return result

    @staticmethod
    def slerp(q0s: np.ndarray, q1s: np.ndarray, t: Union[float, np.ndarray], eps: float = 1e-7):
        q0s = q0s.reshape((1, 4)) if q0s.shape == (4,) else q0s
        q1s = q1s.reshape((1, 4)) if q1s.shape == (4,) else q1s
        assert q0s.shape[-1] == 4 and q0s.ndim == 2 and q0s.shape == q1s.shape
        is_ndarray = isinstance(t, np.ndarray)
        assert not is_ndarray or t.size == q0s.shape[0]
        t = t.reshape((-1, 1)) if is_ndarray else t

        # filp by dot
        q0, q1 = MathHelper.flip_quat_pair_by_dot(q0s, q1s)  # (n, 4), (n, 4)
        if np.allclose(q0, q1):
            return q0

        theta = np.arccos(np.sum(q0 * q1, axis=-1))  # (n,)
        res = MathHelper.unit_quat_arr(q0.shape)  # (n, 4)

        small_flag: np.ndarray = np.abs(theta) < eps  # (small,)
        small_idx = np.argwhere(small_flag).flatten()  # (small,)
        t_small = t[small_idx] if is_ndarray else t  # (small, 1) or float
        res[small_idx] = (1.0 - t_small) * q0[small_idx] + t_small * q1[small_idx]  # (small, 4)
        res[small_idx] /= np.linalg.norm(res[small_idx], axis=-1, keepdims=True)  # (small, 4)

        plain_idx: np.ndarray = np.argwhere(~small_flag).flatten()  # (plain,)
        theta_plain = theta[plain_idx, None]  # (plain, 1)
        inv_sin_theta = 1.0 / np.sin(theta_plain)  # (plain, 1)
        t_plain = t[plain_idx] if is_ndarray else t  # (plain, 1) or float
        res[plain_idx] = (np.sin((1.0 - t_plain) * theta_plain) * inv_sin_theta) * q0[plain_idx] + \
                         (np.sin(t_plain * theta_plain) * inv_sin_theta) * q1[plain_idx]  # (plain, 4)

        res = np.ascontiguousarray(res / np.linalg.norm(res, axis=-1, keepdims=True))
        return res

    @staticmethod
    def torch_skew(v):
        '''
        :param v : torch.Tensor [3,1] or [1,3]
        this function will return the skew matrix (cross product matrix) of a vector
        be sure that it has ONLY 3 element
        it can be autograd
        '''
        import torch
        skv = torch.diag(torch.flatten(v)).roll(1, 1).roll(-1, 0)
        return skv - skv.transpose(0, 1)

    @staticmethod
    def cross_mat(v):
        """create cross-product matrix for v

        Args:
            v (torch.Tensor): a vector with shape (..., 3, 1)
        """
        import torch
        mat = torch.stack((
            torch.zeros_like(v[..., 0, :]), -v[..., 2, :], v[..., 1, :],
            v[..., 2, :], torch.zeros_like(v[..., 1, :]), -v[..., 0, :],
            -v[..., 1, :], v[..., 0, :], torch.zeros_like(v[..., 2, :])
        ), dim=-1).view(*v.shape[:-2], 3, 3)

        return mat

    @staticmethod
    def np_skew(v: np.ndarray):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]
                         ], dtype=np.float64)
