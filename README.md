[![DOI](https://zenodo.org/badge/217773694.svg)](https://zenodo.org/badge/latestdoi/217773694)

# Grid cells in RNNs trained to path integrate

Code to reproduce the trained RNN in [**a unified theory for the origin of grid cells through the lens of pattern formation (NeurIPS '19)**](https://papers.nips.cc/paper/9191-a-unified-theory-for-the-origin-of-grid-cells-through-the-lens-of-pattern-formation) and additional analysis described in this [**preprint**](https://www.biorxiv.org/content/10.1101/2020.12.29.424583v1). 


Quick start:

<img src="./docs/poisson_spiking.gif" width="300" align="right">

* [**inspect_model.ipynb**](inspect_model.ipynb):
  Train a model and visualize its hidden unit ratemaps. 
 
* [**main.py**](main.py):
  or, train a model from the command line.
  
* [**pattern_formation.ipynb**](pattern_formation.ipynb):
  Numerical simulations of pattern-forming dynamics.
  
  
Includes:

* [**trajectory_generator.py**](trajectory_generator.py):
  Generate simulated rat trajectories in a rectangular environment.

* [**place_cells.py**](place_cells.py):
  Tile a set of simulated place cells across the training environment. 
  
* [**model.py**](model.py):
  Contains the vanilla RNN model architecture, as well as an LSTM.
  
* [**trainer.py**](model.py):
  Contains model training loop.
  
* [**models/example_trained_weights.npy**](models/example_trained_weights.npy)
  Contains a set of pre-trained weights.

## Running

We recommend creating a virtual environment:

```shell
$ virtualenv env
$ source env/bin/activate
$ pip install --upgrade pip
```

Then, install the dependencies automatically with `pip install -r requirements.txt`
or manually with:

```shell
$ pip install --upgrade numpy==1.17.2
$ pip install --upgrade tensorflow==2.0.0rc2
$ pip install --upgrade scipy==1.4.1
$ pip install --upgrade matplotlib==3.0.3
$ pip install --upgrade imageio==2.5.0
$ pip install --upgrade opencv-python==4.1.1.26
$ pip install --upgrade tqdm==4.36.0
$ pip install --upgrade opencv-python==4.1.1.26
$ pip install --upgrade torch==1.10.0
```

If you want to train your own models, make sure to properly set the default
save directory in `main.py`! 

## Result

![grid visualization](./docs/RNNgrids.png)
