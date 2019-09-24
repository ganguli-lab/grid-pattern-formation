import tensorflow as tf
import os
import numpy as np


class DataManager(object):
    def __init__(self, options, place_cells):
        self.options = options
        self.place_cells = place_cells


    def avoid_wall(self, position, direction, box_width, box_height):
        '''
        Compute distance and angle to nearest wall
        '''
        x = position[:,0]
        y = position[:,1]
        dists = [box_width-x, box_height-y, box_width+x, box_height+y]
        # if self.options['extended_box']:
        #     dists[0] = 5*box_width-x
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4)*np.pi/2
        theta = angles[np.argmin(dists, axis=0)]
        direction = np.mod(direction, 2*np.pi)
        a_wall = direction - theta
        a_wall = np.mod(a_wall + np.pi, 2*np.pi) - np.pi
        
        near_wall = (d_wall < self.border_region)*(np.abs(a_wall) < np.pi/2)
        turn_angle = np.zeros_like(direction)
        turn_angle[near_wall] = np.sign(a_wall[near_wall])*(np.pi/2 - np.abs(a_wall[near_wall]))

        return near_wall, turn_angle


    def generate_trajectory(self, box_width, box_height, batch_size):
        '''Generate a random walk in a rectangular box'''
        # Parameters
        samples = self.options['sequence_length']
        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias 
        self.border_region = 0.03  # meters

        # Initialize variables
        position = np.zeros([batch_size, samples+2, 2])
        head_dir = np.zeros([batch_size, samples+2])
        position[:,0,0] = np.random.uniform(-1.1, 1.1, batch_size)
        position[:,0,1] = np.random.uniform(-1.1, 1.1, batch_size)
        head_dir[:,0] = np.random.uniform(0, 2*np.pi, batch_size)
        velocity = np.zeros([batch_size, samples+2])
        
        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, samples+1])
        # random_vel = np.random.rayleigh(b, [batch_size, samples+1])
        # v = np.random.rayleigh(b, batch_size)
        random_vel = np.abs(np.random.normal(0, b*np.pi/2, [batch_size, samples+1]))
        v = np.abs(np.random.normal(0, b*np.pi/2, batch_size))

        for t in range(samples+1):
            # Update velocity
            v = random_vel[:,t]
            turn_angle = np.zeros(batch_size)

            if not self.options['periodic']:
                # If in border region, turn and slow down
                near_wall, turn_angle = self.avoid_wall(position[:,t], head_dir[:,t], box_width, box_height)
                v[near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt*random_turn[:,t]

            # Take a step
            velocity[:,t] = v*dt
            update = velocity[:,t,None]*np.stack([np.cos(head_dir[:,t]), np.sin(head_dir[:,t])], axis=-1)
            position[:,t+1] = position[:,t] + update

            # Rotate head direction
            head_dir[:,t+1] = head_dir[:,t] + turn_angle

        # Periodic boundaries
        if self.options['periodic']:
            position[:,:,0] = np.mod(position[:,:,0] + box_width, 2*box_width) - box_width
            position[:,:,1] = np.mod(position[:,:,1] + box_height, 2*box_height) - box_height

        head_dir = np.mod(head_dir + np.pi, 2*np.pi) - np.pi # Periodic variable

        traj = {}
        # Input variables
        traj['init_hd'] = head_dir[:,0,None]
        traj['init_x'] = position[:,1,0,None]
        traj['init_y'] = position[:,1,1,None]

        traj['ego_v'] = velocity[:,1:-1]
        ang_v = np.diff(head_dir, axis=-1)
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:,:-1], np.sin(ang_v)[:,:-1]

        # Target variables
        traj['target_hd'] = head_dir[:,1:-1]
        traj['target_x'] = position[:,2:,0]
        traj['target_y'] = position[:,2:,1]

        return traj

    
    def get_generator(self, batch_size=None, box_width=None, box_height=None):
        if not batch_size:
             batch_size = self.options['batch_size']
        if not box_width:
            box_width = self.options['box_width']
        if not box_height:
            box_height = self.options['box_height']
            
        while True:
            traj = self.generate_trajectory(box_width, box_height, batch_size)
            
            v = tf.stack([traj['ego_v']*tf.cos(traj['target_hd']), 
                  traj['ego_v']*tf.sin(traj['target_hd'])], axis=-1)
            # v = tf.cast(v, tf.float32)

            pos = tf.stack([traj['target_x'], traj['target_y']], axis=-1)
            # pos = tf.cast(pos, tf.float32)
            place_outputs = self.place_cells.get_activation(pos)

            init_pos = tf.stack([traj['init_x'], traj['init_y']], axis=-1)
            # init_pos = tf.cast(init_pos, tf.float32)
            init_actv = tf.squeeze(self.place_cells.get_activation(init_pos))

            inputs = (v, init_actv)
        
            yield (inputs, place_outputs)


    def get_test_batch(self, batch_size=None, box_width=None, box_height=None):
        if not batch_size:
             batch_size = self.options['batch_size']
        if not box_width:
            box_width = self.options['box_width']
        if not box_height:
            box_height = self.options['box_height']
            
        traj = self.generate_trajectory(box_width, box_height, batch_size)
        
        v = tf.stack([traj['ego_v']*tf.cos(traj['target_hd']), 
              traj['ego_v']*tf.sin(traj['target_hd'])], axis=-1)
        # v = tf.cast(v, tf.float32)

        pos = tf.stack([traj['target_x'], traj['target_y']], axis=-1)
        # pos = tf.cast(pos, tf.float32)
        place_outputs = self.place_cells.get_activation(pos)

        init_pos = tf.stack([traj['init_x'], traj['init_y']], axis=-1)
        # init_pos = tf.cast(init_pos, tf.float32)
        init_actv = tf.squeeze(self.place_cells.get_activation(init_pos))

        inputs = (v, init_actv)
        
        return (inputs, pos, place_outputs)