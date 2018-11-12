# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from place_cells import PlaceCells


class PlaceCellsTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.place_cells = PlaceCells()
        
    def test_us_range(self):
        self.assertEqual(self.place_cells.us.shape, (256, 2))
        
        self.assertLessEqual(   np.max(self.place_cells.us),  4.5)
        self.assertGreaterEqual(np.min(self.place_cells.us), -4.5)
        
    def test_get_activatoin(self):
        pos = (0.0, 0.0)
        c = self.place_cells.get_activation(pos)
        
        # Check shape == (256,)
        self.assertEqual(c.shape, (256,))
        
        # Check whether the sum equals to 1
        self.assertAlmostEqual(np.sum(c), 1.0, places=5)

        self.assertLessEqual(   np.max(c),  1.0)
        self.assertGreaterEqual(np.min(c),  0.0)
        
    def test_get_nearest_cell_pos(self):
        pos = np.copy(self.place_cells.us[0])
        c = self.place_cells.get_activation(pos)
        nearest_cell_pos = self.place_cells.get_nearest_cell_pos(c)
        
        self.assertTrue(np.allclose(pos, nearest_cell_pos))
        
        
if __name__ == '__main__':
    unittest.main()
