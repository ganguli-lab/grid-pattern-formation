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
        # self.loss_fun = tf.keras.metrics.categorical_crossentropy
        self.loss_fun = tf.nn.softmax_cross_entropy_with_logits
        self.acc_fun = tf.keras.metrics.categorical_accuracy
        lr = self.options['learning_rate']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss = []
        self.acc = []

        # Set up checkpoints
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=model)
        ckpt_dir = options['save_dir'] + '/' + options['run_ID'] + '/ckpts'
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=100)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


    def compute_loss(self, labels, preds):
        '''Compute cross-entropy loss'''
        loss = tf.reduce_mean(self.loss_fun(labels, preds))
        acc = tf.reduce_mean(self.acc_fun(labels, preds))
        return loss, acc


    def train_step(self, inputs, pc_outputs):
        with tf.GradientTape() as tape:
            preds = self.model(inputs)
            loss, acc = self.compute_loss(pc_outputs, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss, acc


    def train(self, n_epochs=10, n_steps=100, save=True):
        # Construct generator
        gen = self.data_manager.get_generator()

        for epoch in tqdm(range(n_epochs)):
            t = tqdm(range(n_steps), leave=True)
            for _ in t:
                inputs, pc_outputs = next(gen)
                loss, acc = self.train_step(inputs, pc_outputs)
                self.loss.append(loss)
                self.acc.append(acc)
                
                t.set_description('Acc = ' + str(np.round(100*acc, 1)) + '%')

                self.ckpt.step.assign_add(1)

            if save:
                # Save checkpoint
                self.ckpt_manager.save()
                tot_step = self.ckpt.step.numpy()

                # Save a picture of rate maps
                save_ratemaps(self.model, self.data_manager, self.options, step=tot_step)

            