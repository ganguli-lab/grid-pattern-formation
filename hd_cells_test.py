# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from hd_cells import HDCells


class HDCellsTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.hd_cells = HDCells()
        
    def test_us_range(self):
        self.assertEqual(self.hd_cells.us.shape, (12,))

        self.assertLessEqual(   np.max(self.hd_cells.us),  np.pi)
        self.assertGreaterEqual(np.min(self.hd_cells.us), -np.pi)
        
    def test_get_activatoin(self):
        for i in range(10+1):
            angle = 0.1 * i * 2.0 * np.pi - np.pi
            h = self.hd_cells.get_activation(angle)
            
            # Check shape == (12,)
            self.assertEqual(h.shape, (12,))
        
            # Check whether the sum equals to 1
            self.assertAlmostEqual(np.sum(h), 1.0, places=5)
        
            self.assertLessEqual(   np.max(h),  1.0)
            self.assertGreaterEqual(np.min(h),  0.0)
            
            
if __name__ == '__main__':
    unittest.main()
