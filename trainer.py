# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class Trainer(object):
    def __init__(self, model, flags):
        self.model = model
        self._prepare_optimizer(flags)
        self._prepare_summary()
        self.flags=flags

    def _prepare_optimizer(self, flags):
        with tf.variable_scope("opt"):
            output_vars = tf.trainable_variables("model/outputs")

            # Apply L2 regularization to output linear layers
            l2_out = tf.add_n([
                tf.nn.l2_loss(v) for v in output_vars
                if 'bias' not in v.name
            ]) * flags.l2_reg

            optimizer = tf.train.AdamOptimizer(
                learning_rate=flags.learning_rate,
            )
            
            # Nonnegativity constraint on g (breaking the g -> -g symmetry)
            nonneg_g = -tf.reduce_sum(tf.minimum(self.model.g, 0)) * flags.nonneg_obj

            # # l2 constraint on g
            # l2_g = tf.nn.l2_loss(self.model.g) * flags.l2_reg
        
            # l2 constraint on input weights
            l2_win = tf.nn.l2_loss(tf.trainable_variables('model/dense/kernel'))
            l2_win += tf.nn.l2_loss(tf.trainable_variables('model/dense_1/kernel'))

            # Frobenius norm on recurrent weights
            Jall = tf.trainable_variables('model/rnn/basic_rnn_cell/kernel')[0]
            J = tf.gather(Jall, np.arange(flags.rnn_size)+3)
            frob_loss = tf.trace(tf.matmul(J, tf.transpose(J))) * flags.frobenius

            total_loss = self.model.place_loss + \
                self.model.hd_loss + \
                l2_out + nonneg_g  \
                + frob_loss
            
            # Compute gradients
            gvs = optimizer.compute_gradients(total_loss)

            clipped_gvs = []
            for grad, var in gvs:
                if "model/outputs" in var.name:
                    gv = (tf.clip_by_value(grad,
                                           -flags.gradient_clipping,
                                           flags.gradient_clipping), var)
                # elif "basic_rnn_cell/kernel" in var.name:
                #     gv = (tf.clip_by_value(grad, 0, 0), var)
                else:
                    gv = (grad, var)
                clipped_gvs.append(gv)
                
            self.train_op = optimizer.apply_gradients(clipped_gvs)
    
            # # Only train velocity weights
            # J = tf.trainable_variables('model/rnn/basic_rnn_cell/kernel')
            # mask = np.zeros([flags.rnn_size+3, flags.rnn_size])
            # mask[:3] = 1
            # [(grad, var)] = optimizer.compute_gradients(total_loss, var_list=[J])
            # self.train_op = optimizer.apply_gradients([(grad*mask, var)])

    def _prepare_summary(self):
        with tf.name_scope("logs"):
            tf.summary.scalar("place_loss", self.model.place_loss)
            # tf.summary.scalar("hd_loss", self.model.hd_loss)
            tf.summary.scalar("place_accuracy", self.model.place_accuracy)
            # tf.summary.scalar("hd_accuracy", self.model.hd_accuracy)
        self.summary_op = tf.summary.merge_all()

    def train(self, sess, summary_writer, data_manager, step, flags, box_width=None, box_height=None):
        if not box_width:
            box_width = flags.box_width
        if not box_height:
            box_height = flags.box_height
            
        # Train a batch
        if flags.train_or_test=='test':
            feed_dict = data_manager.feed_dict(box_width, box_height)
            _, summary_str = sess.run([self.train_op, self.summary_op], feed_dict=feed_dict)
        else:
            _, summary_str = sess.run([self.train_op, self.summary_op])

        if step % 10 == 0:
            summary_writer.add_summary(summary_str, step)

