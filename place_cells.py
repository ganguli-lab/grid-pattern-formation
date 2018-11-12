# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class PlaceCells(object):
    def __init__(self, cell_size=256, pos_range_min=-1.1, pos_range_max=1.1):
        self.cell_size = cell_size
        std = 0.01
        self.sigma_sq = std * std
        # Means of the gaussian
        self.us = np.random.rand(cell_size, 2) * (pos_range_max - pos_range_min) + pos_range_min

    def get_activation(self, pos):
        """
        Arguments:
          pos: Float Tuple(2)
        """
        d = self.us - pos
        norm2 = np.linalg.norm(d, ord=2, axis=1)
        c = 1e-5 # for numerical stability
        cs = np.exp( -(norm2 - np.min(norm2)) / (2.0 * self.sigma_sq) + c)
        return cs / np.sum(cs)

    def get_nearest_cell_pos(self, activation):
        index = np.argmax(activation)
        return self.us[index]
