# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from visualize import save_ratemaps
from tqdm.autonotebook import tqdm


class Trainer(object):
    def __init__(self, options, model, data_manager):
        self.options = options
        self.model = model
        self.data_manager = data_manager
        lr = self.options['learning_rate']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=model)
        self.ckpt_dir = options['save_dir'] + '/' + options['run_ID'] + '/ckpts'
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=500)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def train_step(self, inputs, pc_outputs, pos):
        with tf.GradientTape() as tape:
            loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        grads = tape.gradient(loss, self.model.trainable_variables)

        # # Clip gradients
        # clipped_grads = []
        # for grad in grads:
        #     clipped_grads.append(tf.clip_by_value(grad, -1e-5, 1e-5))

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # # Block diagonal
        # A = np.ones([self.options['Ng']//32, self.options['Ng']//32])
        # mask = np.kron(np.eye(32), A)
        # # self.model.RNN.weights[1] = mask * self.model.RNN.weights[1]
        # new_weights = [self.model.RNN.weights[0].numpy(), mask * self.model.RNN.weights[1].numpy()]
        # self.model.RNN.set_weights(new_weights)
        
        return loss, err


    def train(self, n_epochs=10, n_steps=100, save=True):
        # Construct generator
        gen = self.data_manager.get_generator()

        # Save at beginning of training
        if save:
            self.ckpt_manager.save()
            np.save(self.ckpt_dir + '/options.npy', self.options)

        for epoch in tqdm(range(n_epochs)):
            t = tqdm(range(n_steps), leave=True)
            for _ in t:
                inputs, pc_outputs, pos = next(gen)
                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)
                
                # t.set_description('Acc = ' + str(np.round(100*acc, 1)) + '%')
                t.set_description('Error = ' + str(np.int(100*err)) + 'cm')

                self.ckpt.step.assign_add(1)

            if save:
                # Save checkpoint
                self.ckpt_manager.save()
                tot_step = self.ckpt.step.numpy()

                # Save a picture of rate maps
                save_ratemaps(self.model, self.data_manager, self.options, step=tot_step)


    def load_ckpt(self, idx):
        '''Restore model from earlier checkpoint'''
        self.ckpt.restore(self.ckpt_manager.checkpoints[idx])

            