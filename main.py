import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from options import get_options
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN, LSTM
from trainer import Trainer

options = get_options()
place_cells = PlaceCells(options)
if options['RNN_type'] == 'RNN':
	model = RNN(options, place_cells)
elif options['RNN_type'] == 'LSTM':
	model = LSTM(options, place_cells)
trajectory_generator = TrajectoryGenerator(options, place_cells)
trainer = Trainer(options, model, trajectory_generator)


# seq_lengths = np.arange(10,55,5)

# for seq_length in seq_lengths:
# 	options['sequence_length'] = seq_length
# 	data_manager = DataManager(options, place_cells)

# 	trainer.train(n_epochs=10, n_steps=1000)

# 	print('Sequence length: ' + str(seq_length))


trainer.train(n_epochs=300, n_steps=1000)




# trainer.train(n_epochs=options['n_epochs'], n_steps=options['n_steps'])