# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py

# データ内の1エピソードのサイズ
EPISODE_LENGTH = 10000


class DataManager(object):
    POS_RANGE_MAX = 1.1
    POS_RANGE_MIN = -1.1
    
    def __init__(self):
        data = h5py.File('mouse_simulation_data.h5', 'r')
       
        num_train_episodes = 790     #since variance is low, maybe drop this?
        num_test_episodes = 10
        num_train_samples = num_train_episodes*EPISODE_LENGTH
        num_test_samples = num_test_episodes*EPISODE_LENGTH
        self.linear_velocities  = data['ego_v'][:num_train_samples] 
        self.theta_x = data['theta_x'][:num_train_samples] 
        self.theta_y =data['theta_y'][:num_train_samples]
        self.pos_x = data['x'][:num_train_samples]
        self.pos_y = data['y'][:num_train_samples]
        self.head_dir = data['head_dir'][:num_train_samples]
      
        self.test_linear_velocities  = data['ego_v'][num_train_samples:num_train_samples+num_test_samples] # (10000,)
        self.test_theta_x = data['theta_x'][num_train_samples:num_train_samples+num_test_samples]          # (10000,)
        self.test_theta_y = data['theta_y'][num_train_samples:num_train_samples+num_test_samples]          # (10000,)
        self.test_pos_x = data['x'][num_train_samples:num_train_samples+num_test_samples]         # (10000,) -1500~1500
        self.test_pos_y = data['y'][num_train_samples:num_train_samples+num_test_samples]         # (10000,) -1500~1500
        self.test_head_dir = data['head_dir'][num_train_samples:num_train_samples+num_test_samples]        # (10000,) -2pi~2pi
        
    def prepare(self, place_cells, hd_cells):
        data_size_train = self.linear_velocities.shape[0]
        data_size_test = self.test_linear_velocities.shape[0]

        # Prepare train inputs data
        self.inputs = np.empty([data_size_train, 3], dtype=np.float32)
        
        self.inputs[:,0] = self.linear_velocities
        self.inputs[:,1] = self.theta_x
        self.inputs[:,2] = self.theta_y

        # Prepare test inputs data
        self.test_inputs = np.empty([data_size_test, 3], dtype=np.float32)
        
        self.test_inputs[:,0] = self.test_linear_velocities
        self.test_inputs[:,1] = self.test_theta_x
        self.test_inputs[:,2] = self.test_theta_y

        # Prepare outputs data
        self.place_outputs = np.empty([data_size_train, place_cells.cell_size])
        self.hd_outputs    = np.empty([data_size_train, hd_cells.cell_size])

        # Prepare test outputs data
        self.test_place_outputs = np.empty([data_size_test, place_cells.cell_size])
        self.test_hd_outputs    = np.empty([data_size_test, hd_cells.cell_size])
        
        for i in range(data_size_train):
            pos = (self.pos_x[i], self.pos_y[i])
            self.place_outputs[i,:] = place_cells.get_activation(pos)
            self.hd_outputs[i,:]    = hd_cells.get_activation(self.head_dir[i])
            
            if i % 100000 == 0:
                print('preparing place cells ' + str(i) + '/' + str(data_size_train))

        for i in range(data_size_test):
            pos = (self.test_pos_x[i], self.test_pos_y[i])
            self.test_place_outputs[i,:] = place_cells.get_activation(pos)
            self.test_hd_outputs[i,:]    = hd_cells.get_activation(self.test_head_dir[i])

    def get_demo_batch(self, batch_size, sequence_length):
        num_episodes = (self.linear_velocities.shape[0]+1) // EPISODE_LENGTH

        inputs_batch        = np.empty([batch_size,
                                        sequence_length,
                                        self.inputs.shape[1]])
        place_outputs_batch = np.empty([batch_size,
                                        sequence_length,
                                        self.place_outputs.shape[1]])
        hd_outputs_batch    = np.empty([batch_size,
                                        sequence_length,
                                        self.hd_outputs.shape[1]])

        place_init_batch = np.empty([batch_size,
                                    sequence_length,
                                    self.place_outputs.shape[1]])
        hd_init_batch    = np.empty([batch_size,
                                    sequence_length,
                                    self.hd_outputs.shape[1]])

        pos_batch = np.empty([batch_size, sequence_length, 2])
        head_dir_batch = np.empty([batch_size, sequence_length])
        
        for i in range(batch_size):
            episode_index = np.random.randint(0, num_episodes)
            pos_in_episode = np.random.randint(0, EPISODE_LENGTH-(sequence_length+1))
            pos = episode_index * EPISODE_LENGTH + pos_in_episode

            #inputs
            inputs_batch[i,:,:]        = self.inputs[pos:pos+sequence_length,:]
            place_init_batch[i,:,:]   = self.place_outputs[pos:pos+sequence_length,:]
            hd_init_batch[i,:,:]      = self.hd_outputs[pos:pos+sequence_length,:]

            #ground truth
            head_dir_batch[i,:] = self.head_dir[pos:pos+sequence_length]
            pos_batch[i,:,0] = self.pos_x[pos:pos+sequence_length]
            pos_batch[i,:,1] = self.pos_y[pos:pos+sequence_length]

            #outputs
            place_outputs_batch[i,:,:] = self.place_outputs[pos+1:pos+sequence_length+1,:] 
            hd_outputs_batch[i,:,:]    = self.hd_outputs[pos+1:pos+sequence_length+1,:] 



        return inputs_batch, place_outputs_batch, hd_outputs_batch, \
            place_init_batch, hd_init_batch, pos_batch, head_dir_batch


    def get_train_batch(self, batch_size, sequence_length):
        num_episodes = (self.linear_velocities.shape[0]+1) // EPISODE_LENGTH

        inputs_batch        = np.empty([batch_size,
                                        sequence_length,
                                        self.inputs.shape[1]])
        place_outputs_batch = np.empty([batch_size,
                                        sequence_length,
                                        self.place_outputs.shape[1]])
        hd_outputs_batch    = np.empty([batch_size,
                                        sequence_length,
                                        self.hd_outputs.shape[1]])

        place_init_batch = np.empty([batch_size,
                                     self.place_outputs.shape[1]])
        hd_init_batch    = np.empty([batch_size,
                                     self.hd_outputs.shape[1]])
        
        for i in range(batch_size):
            episode_index = np.random.randint(0, num_episodes)
            pos_in_episode = np.random.randint(0, EPISODE_LENGTH-(sequence_length+1))
            pos = episode_index * EPISODE_LENGTH + pos_in_episode
            inputs_batch[i,:,:]        = self.inputs[pos:pos+sequence_length,:]
            place_outputs_batch[i,:,:] = self.place_outputs[pos+1:pos+sequence_length+1,:] 
            hd_outputs_batch[i,:,:]    = self.hd_outputs[pos+1:pos+sequence_length+1,:] 
            place_init_batch[i,:]   = self.place_outputs[pos,:]
            hd_init_batch[i,:]      = self.hd_outputs[pos,:]

        return inputs_batch, place_outputs_batch, hd_outputs_batch, \
            place_init_batch, hd_init_batch

    def get_test_batch(self, batch_size, sequence_length):
        num_episodes = (self.test_linear_velocities.shape[0]+1) // EPISODE_LENGTH

        inputs_batch        = np.empty([batch_size,
                                        sequence_length,
                                        self.inputs.shape[1]])
        place_outputs_batch = np.empty([batch_size,
                                        sequence_length,
                                        self.place_outputs.shape[1]])
        hd_outputs_batch    = np.empty([batch_size,
                                        sequence_length,
                                        self.hd_outputs.shape[1]])

        place_init_batch = np.empty([batch_size,
                                     self.place_outputs.shape[1]])
        hd_init_batch    = np.empty([batch_size,
                                     self.hd_outputs.shape[1]])
        
        for i in range(batch_size):
            episode_index = np.random.randint(0, num_episodes)
            pos_in_episode = np.random.randint(0, EPISODE_LENGTH-(sequence_length+1))
            pos = episode_index * EPISODE_LENGTH + pos_in_episode
            inputs_batch[i,:,:]        = self.test_inputs[pos:pos+sequence_length,:]
            place_outputs_batch[i,:,:] = self.test_place_outputs[pos+1:pos+sequence_length+1,:]
            hd_outputs_batch[i,:,:]    = self.test_hd_outputs[pos+1:pos+sequence_length+1,:]
            
            place_init_batch[i,:]   = self.test_place_outputs[pos,:]
            hd_init_batch[i,:]      = self.test_hd_outputs[pos,:]

        return inputs_batch, place_outputs_batch, hd_outputs_batch, \
            place_init_batch, hd_init_batch


    def get_confirm_index_size(self, batch_size, sequence_length):
        # total episode size (=125)
        num_episodes = (self.linear_velocities.shape[0]+1) // EPISODE_LENGTH
        # sequence size per one episode (=4)
        sequence_per_episode = EPISODE_LENGTH // sequence_length
        return (num_episodes * sequence_per_episode // batch_size) - 1

    def get_confirm_batch(self, batch_size, sequence_length, index, condition='train'):
        num_episodes = (self.linear_velocities.shape[0]+1) // EPISODE_LENGTH
        
        # initialize inputs
        inputs_batch     = np.empty([batch_size,
                                     sequence_length,
                                     self.inputs.shape[1]])
        place_init_batch = np.empty([batch_size,
                                     self.place_outputs.shape[1]])
        hd_init_batch    = np.empty([batch_size,
                                     self.hd_outputs.shape[1]])
        
        # intialize ground truth cells
        place_cells_batch = np.empty([batch_size, sequence_length,
                                     self.place_outputs.shape[1]])
        hd_cells_batch    = np.empty([batch_size, sequence_length,
                                     self.hd_outputs.shape[1]])
        
        # initialize ground truth
        head_dir_batch = np.empty([batch_size, sequence_length])
        place_pos_batch  = np.empty([batch_size, sequence_length, 2])

        sequence_per_episode = EPISODE_LENGTH // sequence_length

        sequence_index = index * batch_size
        
        
        if condition=='train':
            for i in range(batch_size):
                episode_index = sequence_index // sequence_per_episode
                pos_in_episode = (sequence_index % sequence_per_episode) * sequence_length
                pos = episode_index * EPISODE_LENGTH + pos_in_episode
                
                # inputs
                inputs_batch[i,:,:] = self.inputs[pos:pos+sequence_length,:]
                place_init_batch[i,:] = self.place_outputs[pos,:]
                hd_init_batch[i,:]  = self.hd_outputs[pos,:]
                
                # ground truth place/hd cell activations
                place_cells_batch[i,:,:]  = self.place_outputs[pos+1:pos+sequence_length+1,:]
                hd_cells_batch[i,:,:] = self.hd_outputs[pos+1:pos+sequence_length+1,:]
                
                # ground truth hd and positions
                head_dir_batch[i,:] = self.head_dir[pos+1:pos+sequence_length+1]
                place_pos_batch[i,:,0] = self.pos_x[pos+1:pos+sequence_length+1]
                place_pos_batch[i,:,1] = self.pos_y[pos+1:pos+sequence_length+1]
                
                sequence_index += 1
        if condition=='test':
            num_episodes = (self.test_linear_velocities.shape[0]+1) // EPISODE_LENGTH
            for i in range(batch_size):
                episode_index = sequence_index // sequence_per_episode
                pos_in_episode = (sequence_index % sequence_per_episode) * sequence_length
                pos = episode_index * EPISODE_LENGTH + pos_in_episode
                
                #inputs
                inputs_batch[i,:,:] = self.test_inputs[pos:pos+sequence_length,:]
                place_init_batch[i,:] = self.test_place_outputs[pos,:]
                hd_init_batch[i,:]  = self.test_hd_outputs[pos,:]
                
                #ground truth place/hd cell activations
                place_cells_batch[i,:,:] = self.test_place_outputs[pos+1:pos+sequence_length+1,:]
                hd_cells_batch[i,:,:] = self.test_hd_outputs[pos+1:pos+sequence_length+1,:]

                # ground truth hd and positions
                head_dir_batch[i,:] = self.test_head_dir[pos+1:pos+sequence_length+1]
                place_pos_batch[i,:,0] = self.test_pos_x[pos+1:pos+sequence_length+1]
                place_pos_batch[i,:,1] = self.test_pos_y[pos+1:pos+sequence_length+1]
                sequence_index += 1
        
        return inputs_batch, place_init_batch, hd_init_batch, place_pos_batch, head_dir_batch, place_cells_batch, hd_cells_batch
