# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from trainer import Trainer
from model import Model
from data_manager import DataManager
from hd_cells import HDCells
from place_cells import PlaceCells
from options import get_options


class TrainerTest(unittest.TestCase):
    def test_init(self):
        np.random.seed(1)
        
        data_manager = DataManager()
        place_cells = PlaceCells()
        hd_cells = HDCells()
        data_manager.prepare(place_cells, hd_cells)

        sequence_length = 100
        
        model = Model(place_cell_size=place_cells.cell_size,
                      hd_cell_size=hd_cells.cell_size,
                      sequence_length=sequence_length)

        flags = get_options()
        trainer = Trainer(data_manager, model, flags)
        
        #self.assertEqual(trainer.g.get_shape()[1], 512)
        

if __name__ == '__main__':
    unittest.main()
