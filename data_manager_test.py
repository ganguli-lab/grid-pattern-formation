# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import unittest

from data_manager import DataManager
from hd_cells import HDCells
from place_cells import PlaceCells


class DataManagerTest(unittest.TestCase):
    def setUp(self):
        self.data_manager = DataManager()

    def test_value_range(self):
        # Pos range is -4.5 ~ 4.5 (実際は-4.03~4.03あたり)
        self.assertLessEqual(   np.max(self.data_manager.pos_xs),  4.5)
        self.assertGreaterEqual(np.min(self.data_manager.pos_xs), -4.5)
        
        # Angle range is -pi ~ pi
        self.assertLessEqual(   np.max(self.data_manager.angles),  np.pi)
        self.assertGreaterEqual(np.min(self.data_manager.angles), -np.pi)

    def test_data_shape(self):
        # Check data shape
        self.assertEqual(self.data_manager.linear_velocities.shape,  (49999,))
        self.assertEqual(self.data_manager.angular_velocities.shape, (49999,))
        
        self.assertEqual(self.data_manager.angles.shape,             (49999,))
        self.assertEqual(self.data_manager.pos_xs.shape,             (49999,))
        self.assertEqual(self.data_manager.pos_zs.shape,             (49999,))
        
    def test_prepare(self):
        np.random.seed(1)
        place_cells = PlaceCells()
        hd_cells = HDCells()

        self.data_manager.prepare(place_cells, hd_cells)

        # Check inputs shape
        self.assertEqual(self.data_manager.inputs.shape, (49999,3))

        # Check outputs shape
        self.assertEqual(self.data_manager.place_outputs.shape, (49999,256))
        self.assertEqual(self.data_manager.hd_outputs.shape,    (49999,12))

    def test_get_train_batch(self):
        np.random.seed(1)
        place_cells = PlaceCells()
        hd_cells = HDCells()

        self.data_manager.prepare(place_cells, hd_cells)

        batch_size = 10
        sequence_length = 100
        out = self.data_manager.get_train_batch(batch_size, sequence_length)
        inputs_batch, place_outputs_batch, hd_outputs_batch, place_init_batch, hd_init_batch = \
            out

        self.assertEqual(inputs_batch.shape,        (batch_size, sequence_length, 3))
        self.assertEqual(place_outputs_batch.shape, (batch_size, sequence_length, 256))
        self.assertEqual(hd_outputs_batch.shape,    (batch_size, sequence_length, 12))
        
        self.assertEqual(place_init_batch.shape,    (batch_size, 256))
        self.assertEqual(hd_init_batch.shape,       (batch_size, 12))

    def test_get_confirm_batch(self):
        np.random.seed(1)
        place_cells = PlaceCells()
        hd_cells = HDCells()
        
        self.data_manager.prepare(place_cells, hd_cells)

        batch_size = 10
        sequence_length = 100

        index_size = self.data_manager.get_confirm_index_size(batch_size, sequence_length)
        # 49
        self.assertEqual(index_size, (50000 // (sequence_length * batch_size)) - 1)

        index = 0
        out = self.data_manager.get_confirm_batch(batch_size, sequence_length, index)
        inputs_batch, place_init_batch, hd_init_batch, place_pos_batch = out

        self.assertEqual(inputs_batch.shape,        (batch_size, sequence_length, 3))
        self.assertEqual(place_init_batch.shape,    (batch_size, 256))
        self.assertEqual(hd_init_batch.shape,       (batch_size, 12))
        self.assertEqual(place_pos_batch.shape,     (batch_size, sequence_length, 2))

        index = index_size-1
        out = self.data_manager.get_confirm_batch(batch_size, sequence_length, index)
        inputs_batch, place_init_batch, hd_init_batch, place_pos_batch = out
        
        self.assertEqual(inputs_batch.shape,        (batch_size, sequence_length, 3))
        self.assertEqual(place_init_batch.shape,    (batch_size, 256))
        self.assertEqual(hd_init_batch.shape,       (batch_size, 12))
        self.assertEqual(place_pos_batch.shape,     (batch_size, sequence_length, 2))
        
        
if __name__ == '__main__':
    unittest.main()
