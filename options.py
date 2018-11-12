# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_options():
    tf.app.flags.DEFINE_string('f', '', 'kernel')
    tf.app.flags.DEFINE_string("save_dir", "saved", "checkpoints,log,options save directory")
    tf.app.flags.DEFINE_string("run_ID", "seq_2000", "save_ID")
    tf.app.flags.DEFINE_integer("batch_size", 50, "batch size") # 10
    tf.app.flags.DEFINE_integer("sequence_length", 2000, "sequence length")    
    tf.app.flags.DEFINE_integer("steps", 300000, "training steps") #300000
    tf.app.flags.DEFINE_integer("save_interval", 500, "saving interval")
    tf.app.flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
    tf.app.flags.DEFINE_float("momentum", 0.9, "momentum")
    tf.app.flags.DEFINE_float("l2_reg", 1e-5, "weight decay") #1e-5
    tf.app.flags.DEFINE_float("gradient_clipping", 1e-5, "gradient clipping")
    return tf.app.flags.FLAGS

