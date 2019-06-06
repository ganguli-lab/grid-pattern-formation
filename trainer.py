# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class Trainer(object):
    def __init__(self, model, flags):
        self.model = model
        self._prepare_optimizer(flags)
        self._prepare_summary()

    def _prepare_optimizer(self, flags):
        with tf.variable_scope("opt"):
            output_vars = tf.trainable_variables("model/outputs")

            # Apply L2 regularization to output linear layers
            l2_loss = tf.add_n([
                tf.nn.l2_loss(v) for v in output_vars
                if 'bias' not in v.name
            ]) * flags.l2_reg

            optimizer = tf.train.AdamOptimizer(
                learning_rate=flags.learning_rate,
            )
            
            # Nonnegativity constraint on g (breaking the g -> -g symmetry)
            nonneg_g = -tf.reduce_sum(tf.minimum(self.model.g, 0)) * flags.nonneg_obj

            # l2 constraint on g
            l2_g = tf.nn.l2_loss(self.model.g)
        
            # l2 constraint on input weights
            l2_win = tf.nn.l2_loss(tf.trainable_variables('model/dense/kernel'))
            l2_win += tf.nn.l2_loss(tf.trainable_variables('model/dense_1/kernel'))


            total_loss = self.model.place_loss + \
                self.model.hd_loss + \
                l2_loss + nonneg_g
            
            # Compute gradients
            gvs = optimizer.compute_gradients(total_loss)

            clipped_gvs = []
            for grad, var in gvs:
                if "model/outputs" in var.name:
                    gv = (tf.clip_by_value(grad,
                                           -flags.gradient_clipping,
                                           flags.gradient_clipping), var)
                else:
                    gv = (grad, var)
                clipped_gvs.append(gv)
                
            self.train_op = optimizer.apply_gradients(clipped_gvs)
    
#             # Only train velocity weights
#             J = tf.trainable_variables('model/rnn/basic_rnn_cell/kernel')
#             mask = np.zeros([515, 512])
#             mask[:3] = 1
#             [(grad, var)] = optimizer.compute_gradients(total_loss, var_list=[J])
#             self.train_op = optimizer.apply_gradients([(grad*mask, var)])

    def _prepare_summary(self):
        with tf.name_scope("logs"):
            tf.summary.scalar("place_loss", self.model.place_loss)
            tf.summary.scalar("hd_loss", self.model.hd_loss)
            tf.summary.scalar("place_accuracy", self.model.place_accuracy)
            tf.summary.scalar("hd_accuracy", self.model.hd_accuracy)
        self.summary_op = tf.summary.merge_all()

    def train(self, sess, summary_writer, step, flags):

        _, summary_str = sess.run([self.train_op, self.summary_op])

        if step % 10 == 0:
            summary_writer.add_summary(summary_str, step)
