# -*- coding: utf-8 -*-
import tensorflow as tf

from data_manager import DataManager, MetaDataManager, get_test_batch
from place_cells import PlaceCells
from hd_cells import HDCells


class Model(object):
    def __init__(self, flags):
        with tf.variable_scope("model"):

            # Prepare trajectories
            if flags.train_or_test == 'train':
                # When training, load data from TFRecord files
                if flags.meta:
                    data_manager = MetaDataManager(flags)
                else:
                    data_manager = DataManager(flags)
                batch = data_manager.get_batch()
            elif flags.train_or_test == 'test':
                # For more flexible testing, load data from feed dicts
                batch = get_test_batch(flags)

            init_x, init_y, init_hd, ego_v, phi_x,  \
                phi_y, target_x, target_y, target_hd = batch

            # # Network must integrate head direction
            # self.inputs = tf.stack([ego_v, phi_x, phi_y], axis=-1)

            # Give network head direction 
            self.inputs = tf.stack([ego_v*tf.cos(target_hd), ego_v*tf.sin(target_hd)], axis=-1)

            self.init_pos = tf.stack([init_x, init_y], axis=-1)
            self.target_pos = tf.stack([target_x, target_y], axis=-1)
            self.target_hd = tf.expand_dims(target_hd, axis=-1)

            place_cells = PlaceCells(
                n_cells=flags.num_place_cells,
                std=flags.place_cell_rf,
                pos_min=-flags.env_size,
                pos_max=flags.env_size,
                DoG = flags.DoG
            )
            hd_cells = HDCells(
                n_cells=flags.num_hd_cells
            )

            place_init = place_cells.get_activation(self.init_pos)
            self.place_init = tf.squeeze(place_init, axis=1)
            hd_init = hd_cells.get_activation(init_hd)
            self.hd_init = tf.squeeze(hd_init, axis=1)
            self.place_outputs = place_cells.get_activation(self.target_pos)
            self.hd_outputs = hd_cells.get_activation(self.target_hd)

            # Drop out probability
            self.keep_prob = tf.constant(flags.keep_prob, dtype=tf.float32)
            
            if flags.RNN_type == 'LSTM':
                self.cell = tf.nn.rnn_cell.LSTMCell(flags.rnn_size, state_is_tuple=True)
            elif flags.RNN_type == 'RNN':
                self.cell = tf.nn.rnn_cell.BasicRNNCell(flags.rnn_size, 
                    activation=tf.keras.layers.Activation(flags.activation))
                

            # init cell
            self.l0 = tf.layers.dense(self.place_init, flags.rnn_size, use_bias=False) + \
                tf.layers.dense(self.hd_init, flags.rnn_size, use_bias=False)

            if flags.RNN_type == 'LSTM':
                # init hidden
                self.m0 = tf.layers.dense(self.place_init, flags.rnn_size, use_bias=False) + \
                    tf.layers.dense(self.hd_init, flags.rnn_size, use_bias=False)

                self.initial_state = tf.nn.rnn_cell.LSTMStateTuple(
                    self.l0,
                    self.m0
                )
            elif flags.RNN_type == 'RNN':
                self.initial_state = self.l0

            self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=self.inputs,
                initial_state=self.initial_state,
                dtype=tf.float32,
                time_major=False
            )

            rnn_output = tf.reshape(self.rnn_output, shape=[-1, flags.rnn_size])
            
            if flags.dense_layer:
                self.g = tf.layers.dense(rnn_output, flags.num_g_cells, use_bias=True)
            else:
                self.g = rnn_output

            g_dropout = tf.nn.dropout(self.g, self.keep_prob)

            with tf.variable_scope("outputs"):
                if flags.place_outputs:
                    place_logits = tf.layers.dense(
                        g_dropout,
                        flags.num_place_cells,
                        use_bias=True
                    )
                    
                    place_outputs_reshaped = tf.reshape(
                        self.place_outputs,
                        shape=[-1, flags.num_place_cells]
                    )

                    self.place_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=place_outputs_reshaped,
                            logits=place_logits
                        )
#                         tf.losses.mean_squared_error(
#                             labels=place_outputs_reshaped,
#                             predictions=place_logits)
                    )
                    
                    self.place_outputs_result = tf.nn.softmax(place_logits)

                    
    
                    self.place_accuracy = tf.reduce_mean(
                        tf.metrics.accuracy(
                            labels=tf.argmax(place_outputs_reshaped, -1),
                            predictions=tf.argmax(place_logits, -1)
                        )
                    )
        
                    if flags.hd_integration:
                        hd_logits = tf.layers.dense(
                            g_dropout,
                            flags.num_hd_cells,
                            use_bias=True
                        )

                        hd_outputs_reshaped = tf.reshape(
                            self.hd_outputs,
                            shape=[-1, flags.num_hd_cells]
                        )
                        self.hd_loss = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=hd_outputs_reshaped,
                                logits=hd_logits
                            )
                        )

                        self.hd_outputs_result = tf.nn.softmax(hd_logits)
                        
                        self.hd_accuracy = tf.reduce_mean(
                            tf.metrics.accuracy(
                                labels=tf.argmax(hd_outputs_reshaped, -1),
                                predictions=tf.argmax(hd_logits, -1)
                            )
                        )
                    else:
                        self.hd_loss = 0
                        self.hd_accuracy = 0
                else:
                    place_logits = tf.layers.dense(
                        g_dropout,
                        2,
                        use_bias=True
                    )
                    
                    place_labels = tf.reshape(self.target_pos, [-1, 2])
                    
                    self.place_loss = tf.losses.mean_squared_error(
                            labels=place_labels,
                            predictions=place_logits
                    )
                    self.hd_loss = 0
                    self.hd_accuracy = 0
                    self.place_accuracy = self.place_loss

                    
              