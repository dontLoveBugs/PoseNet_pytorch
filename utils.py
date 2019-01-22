# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/22 14:26
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import glob
import os
from datetime import datetime
import shutil
import socket

import numpy as np
import torch


# get save dir
def get_output_directory(config):
    if config.resume:
        return os.path.dirname(config.resume)
    else:
        save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        data_name = config.image_path.split('/')[-1]
        save_dir_root = os.path.join(save_dir_root, 'result', data_name)
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

        save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        return save_dir


def get_summary_log_directory(save_path):
    log_path = os.path.join(save_path, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    # if os.path.isdir(log_path):
    #     shutil.rmtree(log_path)  # 递归删除
    # os.makedirs(log_path)
    return log_path


# save checkpoint
def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)


def quat_to_euler(q, is_degree=False):
    """
        convert quaternions to eular angles
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    if is_degree:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)

    return np.array([roll, pitch, yaw])


def array_dist(pred, target):
    return np.linalg.norm(pred - target, 2)


def position_dist(pred, target):
    return np.linalg.norm(pred - target, 2)


def rotation_dist(pred, target):
    pred = quat_to_euler(pred)
    target = quat_to_euler(target)

    return np.linalg.norm(pred - target, 2)


def fit_gaussian(pose_quat):
    # pose_quat = pose_quat.detach().cpu().numpy()

    num_data, _ = pose_quat.shape

    # Convert quat to euler
    pose_euler = []
    for i in range(0, num_data):
        pose = pose_quat[i, :3]
        quat = pose_quat[i, 3:]
        euler = quat_to_euler(quat)
        pose_euler.append(np.concatenate((pose, euler)))

    # Calculate mean and variance
    pose_mean = np.mean(pose_euler, axis=0)
    mat_var = np.zeros((6, 6))
    for i in range(0, num_data):
        pose_diff = pose_euler[i] - pose_mean
        mat_var += pose_diff * np.transpose(pose_diff)

    mat_var = mat_var / num_data
    pose_var = mat_var.diagonal()

    return pose_mean, pose_var
