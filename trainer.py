# -*- coding: utf-8 -*-
import torch
import numpy as np

from visualize import save_ratemaps
import os


class Trainer(object):
    def __init__(self, options, model, trajectory_generator):
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        if os.path.isdir(self.ckpt_dir):
            ckpt = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
            self.model.load_state_dict(torch.load(ckpt))
            print("Restored trained model from {}".format(ckpt))
        else:
            os.mkdir(self.ckpt_dir)
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
        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), err.item()


    def train(self, n_steps=10, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # tbar = tqdm(range(n_steps), leave=False)
        for t in range(n_steps):
            inputs, pc_outputs, pos = next(gen)
            loss, err = self.train_step(inputs, pc_outputs, pos)
            self.loss.append(loss)
            self.err.append(err)

            #Log error rate to progress bar
            # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')

            if save and t%1000==0:
                print('Step {}/{}. Loss: {}. Err: {}cm'.format(
                    t,n_steps,np.round(loss,2),np.round(100*err,2)))
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'iter_{}.pth'.format(t))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                            'most_recent_model.pth'))

                # Save a picture of rate maps
                save_ratemaps(self.model, self.trajectory_generator,
                 self.options, step=t)
            