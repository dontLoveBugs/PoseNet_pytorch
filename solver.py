# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/22 14:38
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from network.posenet import model_parser
from criteria import PoseLoss

from utils import *


class Solver():
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config

        if self.config.resume:
            config, model_dict, optimizer, epoch, best_result = self.load_trained_model()
            self.config = config

            self.model = model_parser(self.config.model, self.config.fixed_weight, self.config.dropout_rate,
                                      self.config.bayesian)
            self.model.load_state_dict(model_dict)

            self.optimizer.load_state_dict(optimizer)
            self.start_epoch = epoch
            self.best_val_loss = best_result

            self.criterion = PoseLoss(self.device, self.config.sx, self.config.sq, self.config.learn_beta)

        else:
            # do not use dropout if not bayesian mode
            if not self.config.bayesian:
                self.config.dropout_rate = 0.0

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            self.model = model_parser(self.config.model, self.config.fixed_weight, self.config.dropout_rate,
                                      self.config.bayesian)

            self.criterion = PoseLoss(self.config.sx, self.config.sq, self.config.learn_beta)
            self.print_network(self.model, self.config.model)

            self.start_epoch = 0
            self.best_train_loss = 10000
            self.best_val_loss = 10000

            self.model_save_path = get_output_directory(config)
            self.summary_save_path = get_summary_log_directory(self.model_save_path)

            if self.config.learn_beta:
                self.optimizer = optim.Adam([{'params': self.model.parameters()},
                                             {'params': [self.criterion.sx, self.criterion.sq]}],
                                            lr=self.config.lr, weight_decay=self.config.weight_decay)
            else:
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=self.config.lr, weight_decay=self.config.weight_decay)

            # write training parameters to config file
            config_txt = os.path.join(self.model_save_path, 'config.txt')
            if not os.path.exists(config_txt):
                with open(config_txt, 'w') as txtfile:
                    args_ = vars(self.config)
                    args_str = ''
                    for k, v in args_.items():
                        args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
                    txtfile.write(args_str)

    def load_trained_model(self):

        checkpoint = torch.load(self.config.resume)
        config = checkpoint['config']
        model_dict = checkpoint['state_dict']
        epoch = checkpoint['epoch']
        optimizer = checkpoint['optimizer']
        best_result = checkpoint['best_result']

        print('Load pretrained network: ', self.config.resume)
        return config, model_dict, epoch, optimizer, best_result

    def print_network(self, model, name):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        print('*' * 20)
        print(name)
        print(model)
        print('*' * 20)

    def loss_func(self, input, target):
        diff = torch.norm(input - target, dim=1)
        diff = torch.mean(diff)
        return diff

    def train(self):

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.config.num_epochs_decay, gamma=0.1)

        num_epochs = self.config.num_epochs

        # Setup for tensorboard
        use_tensorboard = self.config.use_tensorboard
        if use_tensorboard:
            if not os.path.exists(self.summary_save_path):
                os.makedirs(self.summary_save_path)
            writer = SummaryWriter(log_dir=self.summary_save_path)

        since = time.time()

        start_epoch = self.start_epoch
        n_iter = start_epoch * len(self.data_loader['train'])

        # Pre-define variables to get the best model
        best_train_loss = self.best_train_loss
        best_val_loss = self.best_val_loss
        best_train_model = None
        best_val_model = None

        for epoch in range(start_epoch, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 20)

            error_train = []
            error_val = []

            for phase in ['train', 'val']:

                if phase == 'train':
                    scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()

                data_loader = self.data_loader[phase]

                for i, (inputs, poses) in enumerate(data_loader):

                    inputs = inputs.to(self.device)
                    poses = poses.to(self.device)

                    # Zero the parameter gradient
                    self.optimizer.zero_grad()

                    # forward
                    pos_out, ori_out, _ = self.model(inputs)

                    pos_true = poses[:, :3]
                    ori_true = poses[:, 3:]

                    ori_out = F.normalize(ori_out, p=2, dim=1)
                    ori_true = F.normalize(ori_true, p=2, dim=1)

                    loss, _, _ = self.criterion(pos_out, ori_out, pos_true, ori_true)
                    loss_print = self.criterion.loss_print[0]
                    loss_pos_print = self.criterion.loss_print[1]
                    loss_ori_print = self.criterion.loss_print[2]

                    if use_tensorboard:
                        if phase == 'train':
                            error_train.append(loss_print)
                            writer.add_scalar('loss/overall_loss', loss_print, n_iter)
                            writer.add_scalar('loss/position_loss', loss_pos_print, n_iter)
                            writer.add_scalar('loss/rotation_loss', loss_ori_print, n_iter)
                            if self.config.learn_beta:
                                writer.add_scalar('param/sx', self.criterion.sx.item(), n_iter)
                                writer.add_scalar('param/sq', self.criterion.sq.item(), n_iter)

                        elif phase == 'val':
                            error_val.append(loss_print)

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        n_iter += 1

                    print('{}th {} Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format(i, phase,
                                                                                                       loss_print,
                                                                                                       loss_pos_print,
                                                                                                       loss_ori_print))
            # save trained model for each epoch

            error_train_loss = np.median(error_train)
            error_val_loss = np.median(error_val)

            if (epoch + 1) % self.config.model_save_step == 0:
                is_best = error_val_loss < best_val_loss
                save_checkpoint({
                    'config': self.config,
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'best_result': best_val_loss,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, epoch, self.model_save_path)

            if error_train_loss < best_train_loss:
                best_train_loss = error_train_loss
                best_train_model = epoch
            if error_val_loss < best_val_loss:
                best_val_loss = error_val_loss
                best_val_model = epoch

                save_checkpoint({
                    'config': self.config,
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'best_result': best_val_loss,
                    'optimizer': self.optimizer.state_dict(),
                }, True, epoch, self.model_save_path)

            print('Train and Validaion error {} / {}'.format(error_train_loss, error_val_loss))

            if use_tensorboard:
                writer.add_scalars('loss/trainval', {'train': error_train_loss, 'val': error_val_loss}, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def test(self):
        f = open(self.summary_save_path + '/test_result.csv', 'w')

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.config.test_model is None:
            test_model_path = self.model_save_path + '/best_net.pth'
        else:
            test_model_path = self.model_save_path + '/{}_net.pth'.format(self.config.test_model)

        print('Load trained model: ', test_model_path)
        self.model.load_state_dict(torch.load(test_model_path))

        total_pos_loss = 0
        total_ori_loss = 0
        pos_loss_arr = []
        ori_loss_arr = []
        true_pose_list = []
        estim_pose_list = []
        if self.config.bayesian:
            pred_mean = []
            pred_var = []

        num_data = len(self.data_loader)

        for i, (inputs, poses) in enumerate(self.data_loader):
            print(i)

            inputs = inputs.to(self.device)

            # forward
            if self.config.bayesian:
                num_bayesian_test = 100
                pos_array = torch.Tensor(num_bayesian_test, 3)
                ori_array = torch.Tensor(num_bayesian_test, 4)

                for i in range(num_bayesian_test):
                    pos_single, ori_single, _ = self.model(inputs)
                    pos_array[i, :] = pos_single
                    ori_array[i, :] = F.normalize(ori_single, p=2, dim=1)

                pose_quat = torch.cat((pos_array, ori_array), 1).detach().cpu().numpy()
                pred_pose, pred_var = fit_gaussian(pose_quat)

                pos_var = np.sum(pred_var[:3])
                ori_var = np.sum(pred_var[3:])

                pos_out = pred_pose[:3]
                ori_out = pred_pose[3:]
            else:
                pos_out, ori_out, _ = self.model(inputs)
                pos_out = pos_out.squeeze(0).detach().cpu().numpy()
                ori_out = F.normalize(ori_out, p=2, dim=1)
                ori_out = quat_to_euler(ori_out.squeeze(0).detach().cpu().numpy())
                print('pos out', pos_out)
                print('ori_out', ori_out)

            pos_true = poses[:, :3].squeeze(0).numpy()
            ori_true = poses[:, 3:].squeeze(0).numpy()

            ori_true = quat_to_euler(ori_true)
            print('pos true', pos_true)
            print('ori true', ori_true)
            loss_pos_print = array_dist(pos_out, pos_true)
            loss_ori_print = array_dist(ori_out, ori_true)

            true_pose_list.append(np.hstack((pos_true, ori_true)))

            if loss_pos_print < 20:
                estim_pose_list.append(np.hstack((pos_out, ori_out)))

            print(pos_out)
            print(pos_true)

            total_pos_loss += loss_pos_print
            total_ori_loss += loss_ori_print

            pos_loss_arr.append(loss_pos_print)
            ori_loss_arr.append(loss_ori_print)

            if self.config.bayesian:
                print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))
                print('{}th std: pos / ori', pos_var, ori_var)
                f.write('{},{},{},{}\n'.format(loss_pos_print, loss_ori_print, pos_var, ori_var))

            else:
                print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))

        position_error = np.median(pos_loss_arr)
        rotation_error = np.median(ori_loss_arr)

        print('=' * 20)
        print('Overall median pose errer {:.3f} / {:.3f}'.format(position_error, rotation_error))
        print('Overall average pose errer {:.3f} / {:.3f}'.format(np.mean(pos_loss_arr), np.mean(ori_loss_arr)))
        f.close()

        if self.config.save_result:
            f_true = self.summary_save_path + '/pose_true.csv'
            f_estim = self.summary_save_path + '/pose_estim.csv'
            np.savetxt(f_true, true_pose_list, delimiter=',')
            np.savetxt(f_estim, estim_pose_list, delimiter=',')
