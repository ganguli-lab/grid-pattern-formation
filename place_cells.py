# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class PlaceCells(object):
    def __init__(self, n_cells=256, std=0.01, pos_min=-1.1, pos_max=1.1):
        self.n_cells = n_cells
        self.sigma_sq = std * std

        # Place cell means
        grid_x, grid_y = np.mgrid[pos_min:pos_max:16j, pos_min:pos_max:16j]
        self.us = np.stack([grid_x.ravel(), grid_y.ravel()]).T
#         n_cells = np.int(np.sqrt(n_cells))
#         x, y = np.unravel_index(np.arange(n_cells**2), [n_cells, n_cells])
#         means = np.stack([x, y], axis=1)
#         self.us = means * (pos_max - pos_min) / n_cells + pos_min

    def get_activation(self, pos):
        """
        Returns place cell outputs for an input trajectory
        """
        d = pos[:, :, tf.newaxis, :] - self.us[np.newaxis, np.newaxis, ...]
        norm2 = tf.reduce_sum(d**2, axis=-1)
        unnor_logpdf = -(norm2) / (2.0 * self.sigma_sq)
        return tf.nn.softmax(unnor_logpdf)

    def get_nearest_cell_pos(self, activation):
        index = np.argmax(activation)
        return self.us[index]
