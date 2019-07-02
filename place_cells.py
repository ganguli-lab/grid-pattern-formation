# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class PlaceCells(object):
    def __init__(self, n_cells=256, std=0.01,
                 box_width=1.1, box_height=1.1,
                 DoG=False, periodic=False):
        self.n_cells = n_cells
        self.sigma_sq = std * std
        self.DoG = DoG
        self.box_width = box_width
        self.box_height = box_height
        self.periodic = periodic

        # # Place cells on a grid
        # coords = np.linspace(-box_width, box_width, int(np.sqrt(n_cells)))
        # grid_x, grid_y = np.meshgrid(coords, coords)
        # self.us = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # # Random place cell means
        # r = np.random.RandomState(seed=300)
        # usx = r.uniform(-box_width, box_width, n_cells)
        # usy = r.uniform(-box_height, box_height, n_cells)
        # self.us = np.stack([usx, usy], axis=-1)

        # Random place cell means
        usx = tf.random_uniform((n_cells,), -box_width, box_width, seed=300)
        usy = tf.random_uniform((n_cells,), -box_height, box_height, seed=301)
        self.us = tf.Variable(tf.stack([usx, usy], axis=-1))

    def get_activation(self, pos):
        """
        Returns place cell outputs for an input trajectory
        """
        d = tf.abs(pos[:, :, tf.newaxis, :] - self.us[np.newaxis, tf.newaxis, ...])
        print(d.shape)

        if self.periodic:
            dx = tf.gather(d, 0, axis=-1)
            dy = tf.gather(d, 1, axis=-1)
            dx = tf.minimum(dx, 2*self.box_width - dx) 
            dy = tf.minimum(dy, 2*self.box_height - dy)
            d = tf.stack([dx,dy], axis=-1)

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
        index = tf.argmax(activation, axis=-1)
        return tf.gather(self.us, index)
