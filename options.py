# -*- coding: utf-8 -*-
import tensorflow as tf
import os


def get_options():

    options = {}
    options["save_dir"] = "/data3/bsorsch/grid_cell_models"
    options["n_epochs"] = 100
    options["n_steps"] = 500
    options["batch_size"] = 200
    options["sequence_length"] = 10
    options["learning_rate"] = 1e-3
    options["Np"] = 512               # number of place cells
    options["Ng"] = 4096               # number of grid cells
    options["place_cell_rf"] = 0.2 # width of place cell tuning curve
    options["surround_width"] = 2   # factor multiplying with of center
    options["RNN_type"] = "RNN"
    options["activation"] = "relu"
    options["DoG"] = True
    options["periodic"] = False         # periodic boundary conditions
    options["box_width"] = 1.1
    options["box_height"] = 1.1
    run_ID = generate_run_ID(options)
    options["run_ID"] = run_ID

    return options


def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in 
    params.npy file.
    '''
    params = [
        'steps', str(options['sequence_length']),
        'batch', str(options['batch_size']),
        options['RNN_type'],
        str(options['Ng']),
        options['activation'],
        'rf', str(options['place_cell_rf']),
        'DoG', str(options['DoG']),
        'periodic', str(options['periodic']),
        'lr', str(options['learning_rate']),
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')

    return run_ID
