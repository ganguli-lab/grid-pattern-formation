# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, SimpleRNN
from tensorflow.keras.models import Model


class RNN(Model):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options['Ng']
        self.Np = options['Np']
        self.sequence_length = options['sequence_length']
        self.nonneg_reg = options['nonneg_reg']
        self.place_cells = place_cells

        self.encoder = Dense(self.Ng, name='encoder', use_bias=False)
        self.RNN = SimpleRNN(self.Ng, 
                             return_sequences=True,
                             # activation=tf.keras.layers.ReLU(),
                             activation=tf.keras.layers.Activation(options['activation']),
                             recurrent_initializer='glorot_uniform',
                             # recurrent_initializer=tf.keras.initializers.RandomUniform(
                             #    -1/np.sqrt(self.Ng), 1/np.sqrt(self.Ng)),
                             name='RNN',
                             use_bias=False)
        self.decoder = Dense(self.Np, name='decoder', use_bias=False)

        # Loss function
        self.loss_fun = tf.nn.softmax_cross_entropy_with_logits
    
    def g(self, inputs):
        '''Compute grid cell activations'''
        v, P0 = inputs
        init_state = self.encoder(P0)
        g = self.RNN(v, initial_state=init_state)
        return g
    
    def call(self, inputs):
        '''Predict place cell code'''
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos):
        '''Compute loss and decoding error'''
        g = self.g(inputs)
        preds = self.decoder(g)
        loss = tf.reduce_mean(self.loss_fun(pc_outputs, preds))

        # Nonneg reg
        # loss -= self.nonneg_reg * tf.reduce_sum(tf.minimum(g,0))
        # loss += 1e-4 * self.nonneg_reg * tf.reduce_sum(g**2) 

        # # Weight regularization 
        loss += self.nonneg_reg * tf.reduce_sum(self.RNN.weights[1]**2)     # singular value penalty
        # loss += self.nonneg_reg * tf.reduce_sum(tf.abs(self.RNN.weights[1]))
        # loss += self.nonneg_reg * tf.reduce_sum(self.RNN.weights[1] * tf.transpose(self.RNN.weights[1]))   # eig penalty

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((pos - pred_pos)**2, axis=-1)))

        return loss, err


class LSTM(Model):
    def __init__(self, options):
        super(LSTM, self).__init__()
        self.Ng = options['Ng']
        self.Np = options['Np']
        self.sequence_length = options['sequence_length']

        self.encoder1 = Dense(self.Ng, name='encoder1')
        self.encoder2 = Dense(self.Ng, name='encoder1')
        self.M = Dense(self.Ng, name='M')
        self.RNN = tf.keras.layers.LSTM(self.Ng, return_sequences=True,
                             activation=options['activation'],
                             recurrent_activation=options['activation'], name='RNN',
                             recurrent_initializer='glorot_uniform')
        self.decoder = Dense(self.Np, name='decoder')
    
    def g(self, inputs):
        '''Compute grid cell activations'''
        v, P0 = inputs
        l0 = self.encoder1(P0)
        m0 = self.encoder2(P0)
        init_state = (l0, m0)
        Mv = self.M(v)
        g = self.RNN(Mv, initial_state=init_state)
        return g
    
    def call(self, inputs):
        '''Predict place cell code'''
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds
    
    