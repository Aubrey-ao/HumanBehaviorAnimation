import numpy as np


__all__ = ["cal_acc"]


def cal_acc(hat: np.ndarray, gt: np.ndarray):
    return np.mean(hat == gt)