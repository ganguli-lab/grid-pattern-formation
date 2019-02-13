# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
            l2_reg_loss = tf.add_n([ tf.nn.l2_loss(v) for v in output_vars
                                     if 'bias' not in v.name ]) * flags.l2_reg
        
            optimizer = tf.train.AdamOptimizer(
                learning_rate=flags.learning_rate,
            )
            
            total_loss = self.model.place_loss + \
                         self.model.hd_loss + \
                         l2_reg_loss

            # Apply gradient clipping
            gvs = optimizer.compute_gradients(total_loss)
            gradient_clipping = flags.gradient_clipping

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

    def _prepare_summary(self):
        with tf.name_scope("logs"):
            tf.summary.scalar("place_loss", self.model.place_loss)
            tf.summary.scalar("hd_loss",    self.model.hd_loss)
            tf.summary.scalar("place_accuracy", self.model.place_accuracy)
            tf.summary.scalar("hd_accuracy", self.model.hd_accuracy)
        self.summary_op = tf.summary.merge_all()

    def train(self, sess, summary_writer, test_summary_writer, step, flags):
        
        _, summary_str = sess.run([self.train_op, self.summary_op])
        
        if step % 10 == 0:
            summary_writer.add_summary(summary_str, step)

            test_summary_str = sess.run(self.summary_op)

            test_summary_writer.add_summary(test_summary_str, step)