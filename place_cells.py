# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class PlaceCells(object):
    def __init__(
        self, n_cells=256, std=0.01, pos_min=-1.1, pos_max=1.1, DOG=False
    ):
        self.n_cells = n_cells
        self.sigma_sq = std * std
        self.DOG = DOG
        self.pos_min = pos_min
        self.pos_max = pos_max

        # # Place cells on a grid
        # coords = np.linspace(pos_min, pos_max, n_cells)
        # grid_x, grid_y = np.meshgrid(coords, coords)
        # self.us = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # # Random place cell means
        r = np.random.RandomState(seed=300)
        self.us = r.uniform(pos_min, pos_max, [n_cells, 2])

    def get_activation(self, pos):
        """
        Returns place cell outputs for an input trajectory
        """
        d1 = tf.abs(pos[:, :, tf.newaxis, :] - self.us[np.newaxis, np.newaxis, ...])
        d = tf.minimum(d1, 2*self.pos_max - d1)  # for periodic boundaries
        norm2 = tf.reduce_sum(d**2, axis=-1)
        unnor_logpdf = -(norm2) / (2.0 * self.sigma_sq)

        outputs = tf.nn.softmax(unnor_logpdf)

        # Difference of gaussians tuning curve
        if self.DOG:
            logpdf2 = -(norm2) / (2.0 * 3 * self.sigma_sq)
            outputs -= tf.nn.softmax(logpdf2)
            outputs += tf.abs(tf.reduce_min(outputs, axis=-1)[..., tf.newaxis])
            outputs /= tf.reduce_sum(outputs, axis=-1)[..., tf.newaxis]

        return outputs

    def get_nearest_cell_pos(self, activation):
        index = np.argmax(activation)
        return self.us[index]
