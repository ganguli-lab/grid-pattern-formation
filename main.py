import numpy as np
import tensorflow as tf
import torch.cuda

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    default='/mnt/fs2/bsorsch/grid_cells/models/',
                    help='directory to save trained models')
parser.add_argument('--n_epochs',
                    default=100,
                    help='number of training epochs')
parser.add_argument('--n_steps',
                    default=1000,
                    help='batches per epoch')
parser.add_argument('--batch_size',
                    default=200,
                    help='number of trajectories per batch')
parser.add_argument('--sequence_length',
                    default=20,
                    help='number of steps in trajectory')
parser.add_argument('--learning_rate',
                    default=1e-4,
                    help='gradient descent learning rate')
parser.add_argument('--Np',
                    default=512,
                    help='number of place cells')
parser.add_argument('--Ng',
                    default=4096,
                    help='number of grid cells')
parser.add_argument('--place_cell_rf',
                    default=0.12,
                    help='width of place cell center tuning curve (m)')
parser.add_argument('--surround_scale',
                    default=2,
                    help='if DoG, ratio of sigma2^2 to sigma1^2')
parser.add_argument('--RNN_type',
                    default='RNN',
                    help='RNN or LSTM')
parser.add_argument('--activation',
                    default='relu',
                    help='recurrent nonlinearity')
parser.add_argument('--weight_decay',
                    default=1e-4,
                    help='strength of weight decay on recurrent weights')
parser.add_argument('--DoG',
                    default=True,
                    help='use difference of gaussians tuning curves')
parser.add_argument('--periodic',
                    default=False,
                    help='trajectories with periodic boundary conditions')
parser.add_argument('--box_width',
                    default=2.2,
                    help='width of training environment')
parser.add_argument('--box_height',
                    default=2.2,
                    help='height of training environment')
parser.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to use for training')

options = parser.parse_args()
options.run_ID = generate_run_ID(options)

print(f'Using device: {options.device}')

place_cells = PlaceCells(options)
if options.RNN_type == 'RNN':
    model = RNN(options, place_cells)
elif options.RNN_type == 'LSTM':
    # model = LSTM(options, place_cells)
    raise NotImplementedError

# Put model on GPU if using GPU
model = model.to(options.device)

trajectory_generator = TrajectoryGenerator(options, place_cells)

trainer = Trainer(options, model, trajectory_generator)

# Train
trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps)
