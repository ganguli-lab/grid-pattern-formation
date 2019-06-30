# -*- coding: utf-8 -*-

import tensorflow as tf


def get_options():
    tf.app.flags.DEFINE_string('f', '', 'kernel')
    tf.app.flags.DEFINE_string("save_dir", "/data3/bsorsch/grid_cell_models", "checkpoints, log, etc")
    tf.app.flags.DEFINE_string("train_or_test", "train", "train/test mode")
    tf.app.flags.DEFINE_integer("batch_size", 200, "batch size")
    tf.app.flags.DEFINE_integer("sequence_length", 10, "sequence length")
    tf.app.flags.DEFINE_integer("steps", 30000000, "training steps")
    tf.app.flags.DEFINE_integer("save_interval", 1000, "saving interval")
    tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout rate")
    tf.app.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
    tf.app.flags.DEFINE_float("l2_reg", 0, "weight decay")
    tf.app.flags.DEFINE_float("gradient_clipping", 1e-5, "gradient clipping")
    tf.app.flags.DEFINE_integer("num_place_cells", 512, "number place cells")
    tf.app.flags.DEFINE_float("place_cell_rf", 0.2, "receptive field")
    tf.app.flags.DEFINE_integer("num_hd_cells", 12, "number hd cells")
    tf.app.flags.DEFINE_float("hd_cell_rf", 20., "hd cell receptive field")
    tf.app.flags.DEFINE_string("RNN_type", "RNN", "recurrent cell type")
    tf.app.flags.DEFINE_string("activation", "relu", "RNN activation func")
    tf.app.flags.DEFINE_float("nonneg_obj", 0, "strength nonneg constraint")
    tf.app.flags.DEFINE_float("frobenius", 0, "low-rank regularization")
    tf.app.flags.DEFINE_bool("meta", False, "perform meta-learning")
    tf.app.flags.DEFINE_bool("place_outputs", True, "use place cell outputs")
    tf.app.flags.DEFINE_bool("hd_integration", False, "perform hd integration")
    tf.app.flags.DEFINE_bool("DoG", True, "difference of gaussians")
    tf.app.flags.DEFINE_bool("dense_layer", False, "include dense layer g")
    tf.app.flags.DEFINE_integer("num_g_cells", 1024, "num grid cells")
    tf.app.flags.DEFINE_integer("rnn_size", 1024, "num units in RNN")
    tf.app.flags.DEFINE_float("env_size", 1.1, "environment size")
    tf.app.flags.DEFINE_string("dataset", '10_step_periodic_new', "filepath")
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
        'dense', str(flags.dense_layer),
        'DoG', str(flags.DoG),
        'nonneg', str(flags.nonneg_obj),
        'lr', str(flags.learning_rate),
        'l2', str(flags.l2_reg),
        'frobenius', str(flags.frobenius),
        'drop', str(1-flags.keep_prob),
        flags.dataset
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')


    # run_ID = 'tune_frobenius_rectangle'

    return run_ID
