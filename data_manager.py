import tensorflow as tf
import os
import numpy as np



class DataManager(object):
    def __init__(self, flags):
        self.flags = flags


    def avoid_wall(self, position, direction, box_width, box_height):
        '''
        Compute distance and angle to nearest wall
        '''
        x = position[:,0]
        y = position[:,1]
        dists = [box_width-x, box_height-y, box_width+x, box_height+y]
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


    def generate_trajectory(self, box_width, box_height):
        '''Generate a random walk in a rectangular box'''

        # Parameters
        samples = self.flags.sequence_length
        batch_size = self.flags.batch_size
        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias 
        self.border_region = 0.03  # meters

        # Initialize variables
        position = np.zeros([batch_size, samples+2, 2])
        head_dir = np.zeros([batch_size, samples+2])
        position[:,0,0] = np.random.uniform(-box_width, box_width, batch_size)
        position[:,0,1] = np.random.uniform(-box_height, box_height, batch_size)
        head_dir[:,0] = np.random.uniform(0, 2*np.pi, batch_size)
        velocity = np.zeros([batch_size, samples+2])
        
        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, samples+1])
        random_vel = np.random.rayleigh(b, [batch_size, samples+1])
        v = np.random.rayleigh(b, batch_size)

        for t in range(samples+1):
            # Update velocity
            v = random_vel[:,t]
            turn_angle = np.zeros(batch_size)

            if not self.flags.periodic:
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
        if self.flags.periodic:
            position[:,:,0] = np.mod(position[:,:,0] + box_width, 2*box_width) - box_width
            position[:,:,1] = np.mod(position[:,:,1] + box_height, 2*box_height) - box_height

        head_dir = np.mod(head_dir + np.pi, 2*np.pi) - np.pi # Periodic variable

        batch = {}
        # Input variables
        batch['init_hd'] = head_dir[:,0,None]
        batch['init_x'] = position[:,1,0,None]
        batch['init_y'] = position[:,1,1,None]

        batch['ego_v'] = velocity[:,1:-1]
        ang_v = np.diff(head_dir, axis=-1)
        batch['phi_x'], batch['phi_y'] = np.cos(ang_v)[:,:-1], np.sin(ang_v)[:,:-1]

        # Target variables
        batch['target_hd'] = head_dir[:,1:-1]
        batch['target_x'] = position[:,2:,0]
        batch['target_y'] = position[:,2:,1]

        return batch


    def get_batch(self):
        """ Returns placeholders for all trajectory quantities """
        init_x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='init_x')
        init_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='init_y')
        init_hd = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='init_hd')
        ego_v = tf.placeholder(dtype=tf.float32, shape=[None, self.flags.sequence_length], name='ego_v')
        phi_x = tf.placeholder(dtype=tf.float32, shape=[None, self.flags.sequence_length], name='phi_x')
        phi_y = tf.placeholder(dtype=tf.float32, shape=[None, self.flags.sequence_length], name='phi_y')
        target_x = tf.placeholder(dtype=tf.float32, shape=[None, self.flags.sequence_length], name='target_x')
        target_y = tf.placeholder(dtype=tf.float32, shape=[None, self.flags.sequence_length], name='target_y')
        target_hd = tf.placeholder(dtype=tf.float32, shape=[None, self.flags.sequence_length], name='target_hd')

        return init_x, init_y, init_hd, ego_v, phi_x, phi_y, target_hd, target_y, target_hd


    def feed_dict(self, box_width=None, box_height=None):
        """ Constructs a feed dict for passing into session """

        if not box_width:
            box_width = self.flags.box_width
        if not box_height:
            box_height = self.flags.box_height

        # Get batch 
        batch = self.generate_trajectory(box_width, box_height)

        feed_dict={
            'model/init_x:0': batch['init_x'],
            'model/init_y:0': batch['init_y'],
            'model/init_hd:0': batch['init_hd'],
            'model/ego_v:0': batch['ego_v'],
            'model/phi_x:0': batch['phi_x'],
            'model/phi_y:0': batch['phi_y'],
            'model/target_x:0': batch['target_x'],
            'model/target_y:0': batch['target_y'],
            'model/target_hd:0': batch['target_hd'],
            'model/box_width:0': box_width,
            'model/box_height:0': box_height
        }

        return feed_dict



class TFDataManager(object):
    def __init__(self, flags):
        self.flags = flags
        root = '/data3/bsorsch/mouse_trajectories'
        basepath = flags.dataset
        base = os.path.join(root, basepath)

        self.filenames = []
        self.num_files = 0
        for file in os.listdir(base):
            if file.endswith('tfrecord'):
                self.filenames.append(os.path.join(root, basepath, file))
                self.num_files += 1
        self.num_traj_per_file = 100

    def parser(self, record):
        """
        Instantiates the ops used to read and parse the data into tensors.
        """
        feature_map = {
            'init_x':
                tf.FixedLenFeature(
                    shape=[1],
                    dtype=tf.float32),
            'init_y':
                tf.FixedLenFeature(
                    shape=[1],
                    dtype=tf.float32),
            'init_hd':
                tf.FixedLenFeature(
                    shape=[1],
                    dtype=tf.float32),
            'ego_v':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'phi_x':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'phi_y':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'target_x':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'target_y':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'target_hd':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
        }
        example = tf.parse_single_example(record, feature_map)
        batch = [
            example['init_x'],
            example['init_y'],
            example['init_hd'],
            example['ego_v'],
            example['phi_x'],
            example['phi_y'],
            example['target_x'],
            example['target_y'],
            example['target_hd']
        ]
        return batch

    def get_batch(self):
        

        # Define dataset
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self.parser)
        num_traj = self.num_files * self.num_traj_per_file
        dataset = dataset.shuffle(buffer_size=num_traj)
        num_epochs = self.flags.steps // self.flags.batch_size
        dataset = dataset.batch(self.flags.batch_size)
        dataset = dataset.repeat()
        

        iterator = dataset.make_one_shot_iterator()

        batch = iterator.get_next()

        return batch
    
    
# class MetaDataManager(object):
#     def __init__(self, flags):
#         self.flags = flags
#         root = '/data3/bsorsch/mouse_trajectories/'
#         basepath = flags.dataset
#         base = os.path.join(root, basepath)

#         self.filenames = []
#         self.num_files = 0
#         for file in os.listdir(base):
#             if file.endswith('tfrecord'):
#                 self.filenames.append(os.path.join(root, basepath, file))
#                 self.num_files += 1
#         self.num_traj_per_file = 100

#     def parser(self, record):
#         """
#         Instantiates the ops used to read and parse the data into tensors.
#         """
#         feature_map = {
#             'init_x':
#                 tf.FixedLenFeature(
#                     shape=[1],
#                     dtype=tf.float32),
#             'init_y':
#                 tf.FixedLenFeature(
#                     shape=[1],
#                     dtype=tf.float32),
#             'init_hd':
#                 tf.FixedLenFeature(
#                     shape=[1],
#                     dtype=tf.float32),
#             'ego_v':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.sequence_length],
#                     dtype=tf.float32),
#             'phi_x':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.sequence_length],
#                     dtype=tf.float32),
#             'phi_y':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.sequence_length],
#                     dtype=tf.float32),
#             'target_x':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.sequence_length],
#                     dtype=tf.float32),
#             'target_y':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.sequence_length],
#                     dtype=tf.float32),
#             'target_hd':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.sequence_length],
#                     dtype=tf.float32),
#             'place_init':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.num_place_cells],
#                     dtype=tf.float32),
#             'hd_init':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.num_hd_cells],
#                     dtype=tf.float32),
#             'place_outputs':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.num_place_cells*
#                           self.flags.sequence_length],
#                     dtype=tf.float32),
#             'hd_outputs':
#                 tf.FixedLenFeature(
#                     shape=[self.flags.num_hd_cells*
#                           self.flags.sequence_length],
#                     dtype=tf.float32),
#         }
#         example = tf.parse_single_example(record, feature_map)
#         batch = [
#             example['init_x'],
#             example['init_y'],
#             example['init_hd'],
#             example['ego_v'],
#             example['phi_x'],
#             example['phi_y'],
#             example['target_x'],
#             example['target_y'],
#             example['target_hd'],
#             example['place_init'],
#             example['hd_init'],
#             example['place_outputs'],
#             example['hd_outputs']
#         ]
#         return batch

#     def get_batch(self):
        
#         datasets = [
#             tf.data.TFRecordDataset(file).map(self.parser).batch(self.flags.batch_size).shuffle(self.flags.batch_size).repeat()
#                     for file in self.filenames
#         ]
#         choice_idxs = tf.cast(np.repeat(np.arange(10), 20), tf.int64)  # remember to make room for visualization
#         choice_dataset = tf.data.Dataset.from_tensor_slices(choice_idxs).repeat()
#         dataset = tf.contrib.data.choose_from_datasets(datasets, choice_dataset)
#         # dataset = dataset
        
# #         # If want to simply shuffle different boxes
# #         dataset = tf.data.TFRecordDataset(self.filenames)
# #         dataset = dataset.map(self.parser)
# #         num_traj = self.num_files * self.num_traj_per_file
# #         dataset = dataset.shuffle(buffer_size=num_traj)
# #         num_epochs = self.flags.steps // self.flags.batch_size
# #         dataset = dataset.batch(self.flags.batch_size)
# #         dataset = dataset.repeat()
        

#         iterator = dataset.make_one_shot_iterator()

#         batch = iterator.get_next()

#         return batch



