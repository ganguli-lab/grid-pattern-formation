# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class PlaceCells(object):
    def __init__(
        self, n_cells=256, std=0.01, pos_min=-1.1, pos_max=1.1, DoG=False
    ):
        self.n_cells = n_cells
        self.sigma_sq = std * std
        self.DoG = DoG
        self.pos_min = pos_min
        self.pos_max = pos_max

        # Place cells on a grid
        coords = np.linspace(pos_min, pos_max, int(np.sqrt(n_cells)))
        grid_x, grid_y = np.meshgrid(coords, coords)
        self.us = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Random place cell means
        # r = np.random.RandomState(seed=300)
        # self.us = r.uniform(pos_min, pos_max, [n_cells, 2])

        # # Periodic rectangle
        # r = np.random.RandomState(seed=300)
        # usx = r.uniform(pos_min, pos_max, n_cells)
        # usy = r.uniform(pos_min+0.2, pos_max-0.2, n_cells)
        # self.us = np.stack([usx, usy], axis=-1)

    def get_activation(self, pos):
        """
        Returns place cell outputs for an input trajectory
        """
        d = tf.abs(pos[:, :, tf.newaxis, :] - self.us[np.newaxis, np.newaxis, ...])

        # periodic boundaries
        # d = tf.minimum(d, 2*self.pos_max - d)  # for periodic boundaries

        # # # periodic rectangle 
        # dx = tf.gather(d, 0, axis=-1)
        # dy = tf.gather(d, 1, axis=-1)
        # dx = tf.minimum(dx, 2*self.pos_max - dx) 
        # dy = tf.minimum(dy, 2*(self.pos_max-0.2) - dy)
        # d = tf.stack([dx,dy], axis=-1)

        norm2 = tf.reduce_sum(d**2, axis=-1)
        unnor_logpdf = -(norm2) / (2.0 * self.sigma_sq)

        outputs = tf.nn.softmax(unnor_logpdf)

        # Difference of gaussians tuning curve
        if self.DoG:
            logpdf2 = -(norm2) / (2.0 * 2 * self.sigma_sq)
            outputs -= tf.nn.softmax(logpdf2)
            outputs += tf.abs(tf.reduce_min(outputs, axis=-1)[..., tf.newaxis])
            outputs /= tf.reduce_sum(outputs, axis=-1)[..., tf.newaxis]

        return outputs

    def get_nearest_cell_pos(self, activation):
        index = np.argmax(activation)
        return self.us[index]
