# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from model import Model


class ModelTest(unittest.TestCase):
    def test_init(self):
        sequence_length = 100
        model = Model(place_cell_size=256,
                      hd_cell_size=12,
                      sequence_length=sequence_length)
        
        self.assertEqual(model.g.get_shape()[1], 512)
        

if __name__ == '__main__':
    unittest.main()
