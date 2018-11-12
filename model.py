# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, place_cell_size, hd_cell_size, sequence_length, gpu):
      gpu = '/device:GPU:' + str(gpu)
      with tf.device(gpu):
        with tf.variable_scope("model"):
            # Inputs
            self.inputs = tf.placeholder(shape=[None, sequence_length, 3],
                                         dtype=tf.float32)
            # Outputs
            self.place_outputs = tf.placeholder(shape=[None, sequence_length, place_cell_size],
                                                dtype=tf.float32)
            self.hd_outputs = tf.placeholder(shape=[None, sequence_length, hd_cell_size],
                                             dtype=tf.float32)
            
            # Initial place and hd cells input
            self.place_init = tf.placeholder(shape=[None, place_cell_size],
                                             dtype=tf.float32)
            self.hd_init = tf.placeholder(shape=[None, hd_cell_size],
                                          dtype=tf.float32)
            # Drop out probability
            self.keep_prob = tf.placeholder(shape=[], dtype=tf.float32)
            
            cell = tf.nn.rnn_cell.BasicLSTMCell(128,
                                                state_is_tuple=True)

            # init cell
            l0 = tf.layers.dense(self.place_init, 128, use_bias=False) + \
                 tf.layers.dense(self.hd_init, 128, use_bias=False)    #turn off biases?
            # init hidden
            m0 = tf.layers.dense(self.place_init, 128, use_bias=False) + \
                 tf.layers.dense(self.hd_init, 128, use_bias=False)   #turn off biases?
            
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(l0, m0)
            
            rnn_output, rnn_state = tf.nn.dynamic_rnn(cell=cell,
                                                      inputs=self.inputs,
                                                      initial_state=initial_state,
                                                      dtype=tf.float32,
                                                      time_major=False)
            
            # rnn_output=(-1,sequence_length,128), rnn_state=((-1,128), (-1,128))
            rnn_output = tf.reshape(rnn_output, shape=[-1, 128])

            self.g = tf.layers.dense(rnn_output, 512, use_bias=True) 

            g_dropout = tf.nn.dropout(self.g, self.keep_prob)
            
            with tf.variable_scope("outputs"):
                place_logits = tf.layers.dense(g_dropout, place_cell_size, use_bias=True)
                hd_logits    = tf.layers.dense(g_dropout, hd_cell_size, use_bias=True)
                
                place_outputs_reshaped = tf.reshape(self.place_outputs,
                                                    shape=[-1, place_cell_size])
                hd_outputs_reshaped = tf.reshape(self.hd_outputs,
                                                 shape=[-1, hd_cell_size])

                self.place_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=place_outputs_reshaped,
                                                            logits=place_logits))
                self.hd_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=hd_outputs_reshaped,
                                                            logits=hd_logits))
                
                self.hd_outputs_result = tf.nn.softmax(hd_logits)
                self.place_outputs_result = tf.nn.softmax(place_logits)
                
                self.hd_accuracy = tf.reduce_mean(tf.metrics.accuracy(labels=tf.argmax(hd_outputs_reshaped,-1),
                                                       predictions=tf.argmax(hd_logits,-1)))
                self.place_accuracy = tf.reduce_mean(tf.metrics.accuracy(labels=tf.argmax(place_outputs_reshaped,-1),
                                                       predictions=tf.argmax(place_logits,-1)))
                
