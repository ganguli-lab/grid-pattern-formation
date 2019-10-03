import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from options import get_options
from place_cells import PlaceCells
from data_manager import DataManager
from model import RNN
from trainer import Trainer

options = get_options()
place_cells = PlaceCells(options)
model = RNN(options, place_cells)
data_manager = DataManager(options, place_cells)
trainer = Trainer(options, model, data_manager)


seq_lengths = np.arange(10,50,5)

for seq_length in seq_lengths:
	options['sequence_length'] = seq_length
	data_manager = DataManager(options, place_cells)

	trainer.train(n_epochs=12, n_steps=500)

	print('Sequence length: ' + str(seq_length))


# trainer.train(n_epochs=50, n_steps=500)




# trainer.train(n_epochs=options['n_epochs'], n_steps=options['n_steps'])