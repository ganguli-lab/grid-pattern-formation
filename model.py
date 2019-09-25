# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, SimpleRNN
from tensorflow.keras.models import Model


class RNN(Model):
    def __init__(self, options):
        super(RNN, self).__init__()
        self.Ng = options['Ng']
        self.Np = options['Np']
        self.sequence_length = options['sequence_length']

        self.encoder = Dense(self.Ng, name='encoder', use_bias=False)
        self.RNN = SimpleRNN(self.Ng, 
                             return_sequences=True,
                             activation=tf.keras.layers.Activation(options['activation']),
                             name='RNN',
                             recurrent_initializer=tf.keras.initializers.RandomUniform(minval=-0.02,maxval=0.02),
                             use_bias=False)
        self.decoder = Dense(self.Np, name='decoder', use_bias=False)
    
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

    def nonneg_reg(self, activities):
        return tf.reduce_sum(tf.nn.relu(-activities)) * 1e3



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
    
    