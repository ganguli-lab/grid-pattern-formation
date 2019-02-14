# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from data_manager import DataManager
import utils


# For nonnegative constraint
from keras.constraints import non_neg


class Model(object):
    def __init__(self, flags):
      with tf.variable_scope("model"):
        
          data_manager = DataManager(flags)
          batch = data_manager.get_batch()
          init_x, init_y, init_hd, ego_v, theta_x,  \
            theta_y, target_x, target_y, target_hd = batch
            
          self.inputs = tf.stack([ego_v, theta_x, theta_y], axis=-1)
          init_pos = tf.concat([init_x, init_y], axis=-1)
          self.target_pos = tf.stack([target_x, target_y], axis=-1)
          self.target_hd = tf.expand_dims(target_hd, axis=-1)
            
          place_cell_ensembles = utils.get_place_cell_ensembles(
              env_size=2.2,
              neurons_seed=8341,
              targets_type='softmax',
              lstm_init_type='softmax',
              n_pc=[flags.num_place_cells],
              pc_scale=[flags.place_cell_rf])

          head_direction_ensembles = utils.get_head_direction_ensembles(
              neurons_seed=8341,
              targets_type='softmax',
              lstm_init_type='softmax',
              n_hdc=[flags.num_hd_cells],
              hdc_concentration=[flags.hd_cell_rf])
            
          initial_conds = utils.encode_initial_conditions(
              init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
          self.place_init, self.hd_init = initial_conds


          ensembles_targets = utils.encode_targets(
              self.target_pos, self.target_hd, place_cell_ensembles, head_direction_ensembles)
          self.place_outputs, self.hd_outputs = ensembles_targets
            

          # Drop out probability
          self.keep_prob = tf.constant(flags.keep_prob, dtype=tf.float32)
          
          self.cell = tf.nn.rnn_cell.LSTMCell(128,
                                              state_is_tuple=True)

          # init cell
          self.l0 = tf.layers.dense(self.place_init, 128, use_bias=False) + \
               tf.layers.dense(self.hd_init, 128, use_bias=False)    #turn off biases?
          # init hidden
          self.m0 = tf.layers.dense(self.place_init, 128, use_bias=False) + \
               tf.layers.dense(self.hd_init, 128, use_bias=False)   #turn off biases?
          
          self.initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.l0, self.m0)
          
          self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(cell=self.cell,
                                                    inputs=self.inputs,
                                                    initial_state=self.initial_state,
                                                    dtype=tf.float32,
                                                    time_major=False)
          
          # rnn_output=(-1,sequence_length,128), rnn_state=((-1,128), (-1,128))
          rnn_output = tf.reshape(self.rnn_output, shape=[-1, 128])

          self.g = tf.layers.dense(rnn_output, 512, use_bias=True)

          g_dropout = tf.nn.dropout(self.g, self.keep_prob)
          
          with tf.variable_scope("outputs"):
              place_logits = tf.layers.dense(g_dropout, flags.num_place_cells, use_bias=True)
              hd_logits    = tf.layers.dense(g_dropout, flags.num_hd_cells, use_bias=True)
              
              place_outputs_reshaped = tf.reshape(self.place_outputs,
                                                  shape=[-1, flags.num_place_cells])
              hd_outputs_reshaped = tf.reshape(self.hd_outputs,
                                               shape=[-1, flags.num_hd_cells])

              self.place_loss = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits_v2(labels=place_outputs_reshaped,
                                                          logits=place_logits))
              self.hd_loss = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits_v2(labels=hd_outputs_reshaped,
                                                          logits=hd_logits))
              
              self.hd_outputs_result = tf.nn.softmax(hd_logits)
              self.place_outputs_result = tf.nn.softmax(place_logits)
              
              self.hd_accuracy = tf.reduce_mean(tf.metrics.accuracy(labels=tf.argmax(hd_outputs_reshaped,-1),
                                                     predictions=tf.argmax(hd_logits,-1)))
              self.place_accuracy = tf.reduce_mean(tf.metrics.accuracy(labels=tf.argmax(place_outputs_reshaped,-1),
                                                     predictions=tf.argmax(place_logits,-1)))
              
