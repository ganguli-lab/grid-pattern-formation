# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class HDCells(object):
    def __init__(self, cell_size=12):
        self.cell_size = cell_size
        self.concentration = 20.0
        self.us = np.random.rand(cell_size) * 2.0 * np.pi - np.pi
        
    def get_activation(self, angle):
        """
        Arguments:
          angle: Float (radian)
        """
        d = self.us - angle
        hs = np.exp(self.concentration * np.cos(d))
        return hs / np.sum(hs)

    def get_nearest_hd(self, activation):
        index = np.argmax(activation)
        return self.us[index]
