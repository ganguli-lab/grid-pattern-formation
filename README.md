# Grid cells in trained recurrent neural networks

Code to reproduce the trained RNN in [**a unified theory for the origin of grid cells through the lens of pattern formation**](https://papers.nips.cc/paper/9191-a-unified-theory-for-the-origin-of-grid-cells-through-the-lens-of-pattern-formation). 

* [**inspect_model.ipynb**](inspect_model.ipynb):
  Train a model and visualize its hidden unit ratemaps. A set of pre-trained weights is saved in .
 
* [**main.py**](main.py):
  or, train a model from the command line.
  
* [**models/example_trained_weights.npy**](models/example_trained_weights.npy)
  a set of pre-trained weights.


```shell
$ virtualenv env
$ source env/bin/activate
$ pip install --upgrade numpy==1.17.2
$ pip install --upgrade tensorflow==2.0.0rc2
$ pip install --upgrade scipy==1.4.1
$ pip install --upgrade matplotlib==3.0.3
$ pip install --upgrade imageio==2.5.0
$ pip install --upgrade opencv-python==4.1.1.26
$ pip install --upgrade tqdm==4.36.0
```

## Result

![grid visualization](./docs/RNNgrids.png)
