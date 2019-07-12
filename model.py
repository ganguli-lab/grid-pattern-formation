# -*- coding: utf-8 -*-
import tensorflow as tf

from place_cells import PlaceCells
from hd_cells import HDCells
from data_manager import TFDataManager


class Model(object):
    def __init__(self, flags):
        with tf.variable_scope("model"):

            if flags.train_or_test=='train':
                # For faster training, use TFRecords dataset.
                data_manager = TFDataManager(flags)
                batch = data_manager.get_batch()
                init_x, init_y, init_hd, ego_v, phi_x,  \
                    phi_y, target_x, target_y, target_hd = batch
                box_width = flags.box_width
                box_height = flags.box_height
            else:
                # For more flexible testing, use placeholders and feed dicts.
                init_x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='init_x')
                init_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='init_y')
                init_hd = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='init_hd')
                ego_v = tf.placeholder(dtype=tf.float32, shape=[None, flags.sequence_length], name='ego_v')
                phi_x = tf.placeholder(dtype=tf.float32, shape=[None, flags.sequence_length], name='phi_x')
                phi_y = tf.placeholder(dtype=tf.float32, shape=[None, flags.sequence_length], name='phi_y')
                target_x = tf.placeholder(dtype=tf.float32, shape=[None, flags.sequence_length], name='target_x')
                target_y = tf.placeholder(dtype=tf.float32, shape=[None, flags.sequence_length], name='target_y')
                target_hd = tf.placeholder(dtype=tf.float32, shape=[None, flags.sequence_length], name='target_hd')

            if flags.hd_integration:
                # Network must integrate head direction
                self.inputs = tf.stack([ego_v, phi_x, phi_y], axis=-1)
            else:
                # Provide network head direction 
                self.inputs = tf.stack([ego_v*tf.cos(target_hd), ego_v*tf.sin(target_hd)], axis=-1)

            self.init_pos = tf.stack([init_x, init_y], axis=-1)
            self.target_pos = tf.stack([target_x, target_y], axis=-1)
            self.target_hd = tf.expand_dims(target_hd, axis=-1)

            # Compute place/hd cell outputs
            self.place_cells = PlaceCells(
                n_cells=flags.num_place_cells,
                std=flags.place_cell_rf,
                box_width=flags.box_width,
                box_height=flags.box_height,
                DoG = flags.DoG,
                periodic = flags.periodic
            )
            hd_cells = HDCells(
                n_cells=flags.num_hd_cells
            )

            place_init = self.place_cells.get_activation(self.init_pos)
            self.place_init = tf.squeeze(place_init, axis=1)
            hd_init = hd_cells.get_activation(init_hd)
            self.hd_init = tf.squeeze(hd_init, axis=1)
            self.place_outputs = self.place_cells.get_activation(self.target_pos)
            self.hd_outputs = hd_cells.get_activation(self.target_hd)

            # Drop out probability
            self.keep_prob = tf.constant(flags.keep_prob, dtype=tf.float32)
            
            if flags.RNN_type == 'LSTM':
                self.cell = tf.nn.rnn_cell.LSTMCell(flags.rnn_size, state_is_tuple=True)
            elif flags.RNN_type == 'RNN':
                if flags.activation=='leaky_relu':
                    self.cell = tf.nn.rnn_cell.BasicRNNCell(flags.rnn_size, 
                        # activation=tf.keras.layers.Activation(flags.activation))
                        activation=tf.keras.layers.LeakyReLU())
                else:
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

                    # ########
                    # def weighted_softmax_cross_entropy(logits, labels, weights):
                    #     q = tf.nn.softmax(logits)
                    #     return -tf.reduce_sum(labels * weights * tf.log(q), axis=-1)


                    # self.place_loss = tf.reduce_mean(
                    #         weighted_softmax_cross_entropy(
                    #             place_logits,
                    #             place_outputs_reshaped,
                    #             self.place_cells.envelope
                    #             )
                    #     )
                    # #########

                    self.place_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=place_outputs_reshaped,
                            logits=place_logits
                        ) 
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

                    
              