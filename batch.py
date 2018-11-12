# -*- coding: utf-8 -*-
import os

batch_sizes = [10, 20, 40, 70, 100]

for b in batch_sizes:
	os.system('python main.py ' + str(b) + " "+ str(0.01))


place_stds = [0.01, 0.04, 0.08, 0.1, 0.3]

for s in place_stds:
	os.system('python main.py ' + str(10) + " " + str(s))