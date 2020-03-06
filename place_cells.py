# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import scipy


class PlaceCells(object):
    def __init__(self, options):
        self.Np = options['Np']
        self.sigma = options['place_cell_rf']
        self.surround_width = options['surround_width']
        self.box_width = options['box_width']
        self.box_height = options['box_height']
        self.is_periodic = options['periodic']
        self.DoG = options['DoG']
        
        # Randomly tile place cell centers
        tf.random.set_seed(0)
        usx = tf.random.uniform((self.Np,), -self.box_width/2, self.box_width/2, dtype=tf.float64)
        usy = tf.random.uniform((self.Np,), -self.box_height/2, self.box_height/2, dtype=tf.float64)
        self.us = tf.stack([usx, usy], axis=-1)

        
    def get_activation(self, pos):
        '''Get place cell activations for a given position'''
        d = tf.abs(pos[:, :, tf.newaxis, :] - self.us[tf.newaxis, tf.newaxis, ...])

        if self.is_periodic:
            dx = tf.gather(d, 0, axis=-1)
            dy = tf.gather(d, 1, axis=-1)
            dx = tf.minimum(dx, self.box_width - dx) 
            dy = tf.minimum(dy, self.box_height - dy)
            d = tf.stack([dx,dy], axis=-1)

        norm2 = tf.reduce_sum(d**2, axis=-1)
        outputs = tf.nn.softmax(-norm2/(2*self.sigma**2))

        if self.DoG:
            outputs -= tf.nn.softmax(-norm2/(2*self.surround_width*self.sigma**2))
            outputs += tf.abs(tf.reduce_min(outputs, axis=-1, keepdims=True))
            outputs /= tf.reduce_sum(outputs, axis=-1, keepdims=True)
        return outputs

    
    def get_nearest_cell_pos(self, activation):
        '''Return location of maximally active place cell'''
        _, idxs = tf.math.top_k(activation, k=3)
        return tf.reduce_mean(tf.gather(self.us, idxs), axis=-2)
        

    def grid_pc(self, pc_outputs, res=32):
        ''' Interpolate place cell outputs onto a grid'''
        coordsx = np.linspace(-self.box_width/2, self.box_width/2, res)
        coordsy = np.linspace(-self.box_height/2, self.box_height/2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        us_np = self.us.numpy()
        pc_outputs = pc_outputs.numpy().reshape(-1, self.Np)
        
        T = pc_outputs.shape[0]
        pc = np.zeros([T, res, res])
        for i in range(len(pc_outputs)):
            gridval = scipy.interpolate.griddata(us_np, pc_outputs[i], grid)
            pc[i] = gridval.reshape([res, res])
        
        return pc

    def compute_covariance(self, res=30):
        '''Compute spatial covariance matrix of place cell outputs'''
        pos = np.array(np.meshgrid(np.linspace(-self.box_width/2, self.box_width/2, res),
                         np.linspace(-self.box_height/2, self.box_height/2, res))).T

        pos = pos.astype(np.float32)

        pc_outputs = self.get_activation(pos)
        pc_outputs = tf.reshape(pc_outputs, (-1, self.Np))

        C = pc_outputs@tf.transpose(pc_outputs)
        Csquare = tf.reshape(C, (res,res,res,res))

        Cmean = np.zeros([res,res])
        for i in range(res):
            for j in range(res):
                Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
                
        Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)

        return Cmean