# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class HDCells(object):
    def __init__(self, n_cells=12):
        self.n_cells = n_cells
        self.concentration = 20.0
        self.us = np.random.rand(n_cells) * 2.0 * np.pi - np.pi

    def get_activation(self, angle):
        """
        Returns hd cell outputs for an input trajectory
        """
        d = angle[:, :, tf.newaxis] - self.us[np.newaxis, :]
        unnor_logpdf = self.concentration * tf.cos(d)
        return tf.nn.softmax(unnor_logpdf)

    def get_nearest_hd(self, activation):
        index = np.argmax(activation)
        return self.us[index]
