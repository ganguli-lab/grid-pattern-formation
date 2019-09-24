# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class PlaceCells(object):
    def __init__(self, n_cells=256, std=0.01,
                 box_width=1.1, box_height=1.1,
                 DoG=False, periodic=False):
        self.n_cells = n_cells
        self.sigma_sq = std**2
        self.DoG = DoG
        self.box_width = box_width
        self.box_height = box_height
        self.periodic = periodic


        # # Vary place cell rfs
        # self.sigma_sq = np.linspace(std**2, 2*std**2, n_cells)

        # # Place cells on a grid
        # coords = np.linspace(-box_width, box_width, int(np.sqrt(n_cells)))
        # grid_x, grid_y = np.meshgrid(coords, coords)
        # usx = grid_x.ravel().astype(np.float32)
        # usy = grid_y.ravel().astype(np.float32)
        # self.us = tf.Variable(np.stack([usx, usy], axis=-1), name='place_cell_us', trainable=False)

        # Random place cell means
        r = np.random.RandomState(seed=300)
        usx = r.uniform(-box_width, box_width, (n_cells,)).astype(np.float32)
        usy = r.uniform(-box_height, box_height, (n_cells,)).astype(np.float32)
        self.us = tf.Variable(np.stack([usx, usy], axis=-1), name='place_cell_us', trainable=False)

        # Random place cell means
        # usx = tf.random_uniform((n_cells,), -box_width, box_width, seed=300)
        # usy = tf.random_uniform((n_cells,), -box_height, box_height, seed=301)
        # self.us = tf.Variable(np.stack([usx, usy], axis=-1), name='place_cell_us', trainable=False)


        # Envelope function 
        r2 = tf.reduce_sum(self.us**2, axis=-1)
        envelope_width = 0.6
        self.envelope = tf.exp(-r2 / envelope_width**2)


    def get_activation(self, pos):
        """
        Returns place cell outputs for an input trajectory
        """
        d = tf.abs(pos[:, :, tf.newaxis, :] - self.us[tf.newaxis, tf.newaxis, ...])

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
#             logpdf2 = -(norm2) / (2.0 * 2 * self.sigma_sq)
            logpdf2 = -(norm2) / (2.0 * 3 * self.sigma_sq)
            outputs -= tf.nn.softmax(logpdf2)
            outputs += tf.abs(tf.reduce_min(outputs, axis=-1)[..., tf.newaxis])
            outputs /= tf.reduce_sum(outputs, axis=-1)[..., tf.newaxis]

        # # Envelope function 
        # r2 = tf.reduce_sum(self.us**2, axis=-1)
        # envelope_width = 0.6
        # envelope = tf.exp(-r2 / envelope_width**2)
        # outputs *= envelope
        # outputs += tf.abs(tf.reduce_min(outputs, axis=-1)[..., tf.newaxis])
        # outputs /= tf.reduce_sum(outputs, axis=-1)[..., tf.newaxis]

        return outputs

    def get_nearest_cell_pos(self, activation):
        ''' Return localtion of maximally activate place cell '''
        index = tf.argmax(activation, axis=-1)
        return tf.gather(self.us, index)


    def update_us(self, box_width, box_heigth):
        ''' If box size changes, update place cell locs accordingly'''
        r = np.random.RandomState(seed=300)
        usx = r.uniform(-box_width, box_width, n_cells)
        usy = r.uniform(-box_height, box_height, n_cells)
        self.us = np.stack([usx, usy], axis=-1)
