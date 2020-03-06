# -*- coding: utf-8 -*-
import tensorflow as tf
import os


def get_options():
    ''' Define training parameters and hyperparameters'''
    options = {}
    options["save_dir"] = "/mnt/fs2/bsorsch/grid_cells/models/"
    options["n_epochs"] = 5
    options["n_steps"] = 1000
    options["batch_size"] = 200
    options["sequence_length"] = 20
    options["learning_rate"] = 1e-4
    options["Np"] = 512              # number of place cells
    options["Ng"] = 4096             # number of grid cells
    options["place_cell_rf"] = 0.12  # width of place cell tuning curve
    options["surround_width"] = 2   # width of place cell surround
    options["RNN_type"] = "RNN"
    options["activation"] = "relu"
    options['nonneg_reg'] = 1e-4
    options["DoG"] = True
    options["periodic"] = False         # periodic boundary conditions
    options["box_width"] = 2.2
    options["box_height"] = 2.2
    run_ID = generate_run_ID(options)
    options["run_ID"] = run_ID

    return options