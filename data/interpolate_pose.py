# coding: utf-8
# Created by luuil@outlook.com at 5/7/2021

import json
import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_pose(key_pose, interp_len: int = 12, interp: str = 'linear'):
    """
    :param key_pose: json format, 42 elements
    :param interp_len: length of pose sequence
    :param interp: interpolate method
    :return: pose sequence
    """
    pose_out_tmp = []
    repeat_num = int(interp_len / 2)
    for pose_name, pose_max in key_pose.items():
        if interp == "linear":
            pose_each = np.linspace(0, pose_max, num=repeat_num).tolist() \
                        + [pose_max] * repeat_num \
                        + np.linspace(pose_max, 0, num=repeat_num).tolist()
            pose_each = np.array(pose_each)
        elif interp == "spline":
            x = [0, repeat_num, interp_len]
            y = [0, pose_max, 0]
            cs = CubicSpline(x, y, bc_type="clamped")
            before_middle = [cs(k) for k in range(0, repeat_num)]
            middle = [pose_max] * repeat_num
            after_middle = [cs(k) for k in range(repeat_num + 1, interp_len + 1)]

            pose_each = before_middle + middle + after_middle
            pose_each = np.array(pose_each)
        else:
            raise ValueError("The name of function is wrong")
        pose_out_tmp.append(pose_each)
    pose_seq = np.array(pose_out_tmp)
    pose_seq = pose_seq.transpose((1, 0))
    return pose_seq


if __name__ == "__main__":
    with open('./pose.json') as f:
        key_poses_dict = json.load(f)
    # key_pose_name = "Haixiu_P0"
    pose_all = list()
    for key_pose_name in key_poses_dict:
        key_pose = key_poses_dict[key_pose_name]
        pose_seq = interpolate_pose(key_pose, interp_len=20)

        pose_seq_transition = np.zeros((10, pose_seq.shape[1]))
        pose_all.append(pose_seq_transition)
        pose_all.append(pose_seq)

        print(f"{key_pose_name}: len={len(pose_seq)}")
        np.save(f"poses_npy/{key_pose_name}.npy", pose_seq)
    pose_all = np.vstack(pose_all)
    np.save("poses_npy/all.npy", pose_all)
