# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from visualize import save_ratemaps
from tqdm import tqdm


class Trainer(object):
    def __init__(self, options, model, trajectory_generator):
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        lr = self.options.learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=model)
        self.ckpt_dir = options.save_dir + '/' + options.run_ID + '/ckpts'
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=500)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored trained model from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing new model from scratch.")


    def train_step(self, inputs, pc_outputs, pos):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        with tf.GradientTape() as tape:
            loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        
        return loss, err


    def train(self, n_epochs=10, n_steps=100, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_epochs: Number of training epochs
            n_steps: Number of batches of trajectories per epoch
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # Save at beginning of training
        if save:
            self.ckpt_manager.save()
            np.save(self.ckpt_dir + '/options.npy', self.options)

        for epoch in tqdm(range(n_epochs)):
            t = tqdm(range(n_steps), leave=False)
            for _ in t:
                inputs, pc_outputs, pos = next(gen)
                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)
                
                #Log error rate
                t.set_description('Error = ' + str(np.int(100*err)) + 'cm')

                self.ckpt.step.assign_add(1)

            if save:
                # Save checkpoint
                self.ckpt_manager.save()
                tot_step = self.ckpt.step.numpy()

                # Save a picture of rate maps
                save_ratemaps(self.model, self.trajectory_generator, self.options, step=tot_step)


    def load_ckpt(self, idx):
        ''' Restore model from earlier checkpoint. '''
        self.ckpt.restore(self.ckpt_manager.checkpoints[idx])

            