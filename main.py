import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from options import get_options
from place_cells import PlaceCells
from data_manager import DataManager
from model import RNN
from trainer import Trainer

options = get_options()
model = RNN(options)
place_cells = PlaceCells(options)
data_manager = DataManager(options, place_cells)
trainer = Trainer(options, model, data_manager)

trainer.train(n_epochs=options['n_epochs'], n_steps=options['n_steps'])