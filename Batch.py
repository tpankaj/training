"""Processes data into batches for training and validation."""
from Parameters import ARGS
from libs.utils2 import z2o
from libs.vis2 import mi
import numpy as np
import torch
import sys
import torch.nn.utils as nnutils
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Batch:

    def clear(self):
        ''' Clears batch variables before forward pass '''
        self.camera_data = torch.FloatTensor().cuda()
        self.metadata = torch.FloatTensor().cuda()
        self.target_data = torch.FloatTensor().cuda()
        self.names = []
        self.outputs = None
        self.loss = None

    def __init__(self, net):
        self.net = net
        self.camera_data = None
        self.metadata = None
        self.target_data = None
        self.names = None
        self.outputs = None
        self.loss = None

    def forward(self, camera_data, metadata, target_data,
                optimizer, criterion, data_moment_loss_record):
        self.camera_data = camera_data.cuda()
        self.metadata = metadata.cuda()
        self.target_data = target_data.cuda()
        optimizer.zero_grad()
        self.outputs = self.net(Variable(self.camera_data),
                                Variable(self.metadata)).cuda()
        self.loss = criterion(self.outputs, Variable(self.target_data))

        for b in range(ARGS.batch_size):
            t = self.target_data[b].cpu().numpy()
            o = self.outputs[b].data.cpu().numpy()
            a = (self.target_data[b] - self.outputs[b].data).cpu().numpy()
            loss = np.sqrt(a * a).mean()
            data_moment_loss_record[(tuple(t), tuple(o))] = loss

    def backward(self, optimizer):
        self.loss.backward()
        nnutils.clip_grad_norm(self.net.parameters(), 1.0)
        optimizer.step()

    def display(self):
        if ARGS.display:
            o = self.outputs[0].data.cpu().numpy()
            t = self.target_data[0].cpu().numpy()

            print(
                'Loss:',
                np.round(
                    self.loss.data.cpu().numpy()[0],
                    decimals=5))
            a = self.camera_data[0][:].cpu().numpy()
            b = a.transpose(1, 2, 0)
            h = np.shape(a)[1]
            w = np.shape(a)[2]
            c = np.zeros((10 + h * 2, 10 + 2 * w, 3))
            c[:h, :w, :] = z2o(b[:, :, 3:6])
            c[:h, -w:, :] = z2o(b[:, :, :3])
            c[-h:, :w, :] = z2o(b[:, :, 9:12])
            c[-h:, -w:, :] = z2o(b[:, :, 6:9])
            mi(c, 'cameras')
            print(a.min(), a.max())
            plt.figure('steer')
            plt.clf()
            plt.ylim(-0.05, 1.05)
            plt.xlim(0, len(t))
            plt.plot([-1, 60], [0.49, 0.49], 'k')  # plot in black
            plt.plot(o, 'og')  # plot using green circle markers
            plt.plot(t, 'or')  # plot using red circle markers
            plt.title(self.names[0])
            plt.pause(sys.float_info.epsilon)
