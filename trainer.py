# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from visualize import save_ratemaps
from tqdm.autonotebook import tqdm


class Trainer(object):
    def __init__(self, options, model, data_manager, place_cells):
        self.options = options
        self.model = model
        self.data_manager = data_manager
        self.place_cells = place_cells
        # self.loss_fun = tf.keras.metrics.categorical_crossentropy
        self.loss_fun = tf.nn.softmax_cross_entropy_with_logits
        self.acc_fun = tf.keras.metrics.categorical_accuracy
        lr = self.options['learning_rate']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=model)
        ckpt_dir = options['save_dir'] + '/' + options['run_ID'] + '/ckpts'
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=100)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


    def compute_loss(self, labels, preds, pos):
        '''Compute cross-entropy loss'''
        loss = tf.reduce_mean(self.loss_fun(labels, preds))
        acc = tf.reduce_mean(self.acc_fun(labels, preds))
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = tf.reduce_mean(tf.sqrt(tf.reduce_sum((pos - pred_pos)**2, axis=-1)))
        # loss += tf.reduce_mean(self.model.RNN.weights[1]**2) * 1
        return loss, err


    def train_step(self, inputs, pc_outputs, pos):
        with tf.GradientTape() as tape:
            preds = self.model(inputs)
            loss, err = self.compute_loss(pc_outputs, preds, pos)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss, err


    def train(self, n_epochs=10, n_steps=100, save=True):
        # Construct generator
        gen = self.data_manager.get_generator()

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

            