# -*- coding: utf-8 -*-

import tensorflow as tf


def get_options():
    tf.app.flags.DEFINE_string('f', '', 'kernel')
    tf.app.flags.DEFINE_string("save_dir", "saved", "checkpoints, log, etc")
    tf.app.flags.DEFINE_string("train_or_test", "train", "train/test mode")
    tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
    tf.app.flags.DEFINE_integer("sequence_length", 50, "sequence length")
    tf.app.flags.DEFINE_integer("steps", 30000000, "training steps")
    tf.app.flags.DEFINE_integer("save_interval", 1000, "saving interval")
    tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout rate")
    tf.app.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
    tf.app.flags.DEFINE_float("l2_reg", 0, "weight decay")
    tf.app.flags.DEFINE_float("gradient_clipping", 1e-5, "gradient clipping")
    tf.app.flags.DEFINE_integer("num_place_cells", 256, "number place cells")
    tf.app.flags.DEFINE_float("place_cell_rf", 0.16, "receptive field")
    tf.app.flags.DEFINE_integer("num_hd_cells", 12, "number hd cells")
    tf.app.flags.DEFINE_float("hd_cell_rf", 20., "hd cell receptive field")
    tf.app.flags.DEFINE_string("RNN_type", "RNN", "recurrent cell type")
    tf.app.flags.DEFINE_string("activation", "relu", "RNN activation func")
    tf.app.flags.DEFINE_float("nonneg_obj", 1e-5, "strength nonneg constraint")
    tf.app.flags.DEFINE_float("frobenius", 0, "low-rank regularization")
    tf.app.flags.DEFINE_bool("meta", False, "perform meta-learning")
    tf.app.flags.DEFINE_bool("place_outputs", True, "use place cell outputs")
    tf.app.flags.DEFINE_bool("hd_integration", False, "perform hd integration")
    tf.app.flags.DEFINE_bool("DOG", True, "difference of gaussians")
    tf.app.flags.DEFINE_bool("dense_layer", False, "include dense layer g")
    tf.app.flags.DEFINE_integer("num_g_cells", 1024, "num grid cells")
    tf.app.flags.DEFINE_integer("rnn_size", 1024, "num units in RNN")
    tf.app.flags.DEFINE_float("env_size", 1.1, "environment size")
    tf.app.flags.DEFINE_string("dataset", '50_step_periodic_new', "filepath")
    run_ID = generate_run_ID(tf.app.flags.FLAGS)
    tf.app.flags.DEFINE_string("run_ID", run_ID, "save_ID")
    return tf.app.flags.FLAGS


def generate_run_ID(flags):
    ''' 
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in 
    params.txt file.
    '''
    params = [
        'steps', str(flags.sequence_length),
        'batch', str(flags.batch_size),
        flags.RNN_type,
        str(flags.rnn_size),
        flags.activation,
        'rf', str(flags.place_cell_rf),
        'DOG', str(flags.DOG),
        'nonneg', str(flags.nonneg_obj),
        'lr', str(flags.learning_rate),
        'l2', str(flags.l2_reg),
        'drop', str(1-flags.keep_prob),
        flags.dataset
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID
