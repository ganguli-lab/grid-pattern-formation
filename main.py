# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

from model import Model
from trainer import Trainer
from options import get_options
from data_manager import DataManager
import visualize
import pickle


from matplotlib import pyplot as plt

flags = get_options()


def load_checkpoints(sess):
    saver = tf.train.Saver(max_to_keep=100)
    checkpoint_dir = flags.save_dir + "/" + flags.run_ID + "/ckpts"

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)

        tokens = checkpoint.model_checkpoint_path.split("-")
        step = int(tokens[-1])
        print(
            "Loaded checkpoint: {0}, step={1}".format(
                checkpoint.model_checkpoint_path, step
            )
        )
        return saver, step + 1
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return saver, 0


def save_checkponts(sess, saver, global_step):
    checkpoint_dir = flags.save_dir + "/" + flags.run_ID + "/ckpts"
    saver.save(
        sess, checkpoint_dir + '/' + 'checkpoint', global_step=global_step
    )
    print("Checkpoint saved")


def save_params(sess):
    """ Save training parameters to file inside checkpoints folder. """
    checkpoint_dir = flags.save_dir + "/" + flags.run_ID

    attrs = dir(flags)
    params = {}
    for i in range(len(attrs)):
        params[attrs[i]] = getattr(flags, attrs[i])
        
    # Save to pickle
    with open(checkpoint_dir + '/params.pkl', 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    # Save to txt
    with open(checkpoint_dir + '/params.txt', 'wb') as f:
        for K, V in params.items():
            f.write(K + "\t" + str(V) + "\n")


def train(
    sess, model, trainer, saver, summary_writer, start_step, data_manager
):
    for i in range(start_step, flags.steps):
        trainer.train(sess, summary_writer, data_manager, step=i, flags=flags)

        if i % flags.save_interval == 0 and i > 1:
            save_checkponts(sess, saver, i)
            save_name = flags.run_ID
            visualize.save_visualization(
                sess, model, save_name, data_manager, step=i - 1, flags=flags
            )

        # if i % (5 * flags.save_interval) == 0 and i > 1:
        #     save_name = flags.run_ID
        #     visualize.save_autocorr(
        #         sess, model, save_name, data_manager, step=i-1, flags=flags
        #     )


def meta_train(
    sess, model, trainer, saver, summary_writer, start_step, data_manager
):
    
    # Initialize with standard box
    box_width = flags.box_width
    box_height = flags.box_height

    for i in range(start_step, flags.steps):

        trainer.train(sess, summary_writer,
                        data_manager, step=i,
                        flags=flags, box_width=box_width,
                        box_height=box_height)

        if i % flags.save_interval == 0:
            save_checkponts(sess, saver, i)
            save_name = flags.run_ID
            visualize.save_visualization(
                sess, model, save_name, data_manager, step=i-1, flags=flags
            )

            # Generate a new training environment
            box_width = np.random.uniform(0.8, 1.2)
            box_height = np.random.uniform(0.8, 1.2)

            #Update place cell centers
            usx = np.random.uniform(-box_width, box_width, flags.num_place_cells).astype(np.float32)
            usy = np.random.uniform(-box_height, box_height, flags.num_place_cells).astype(np.float32)
            plt.clf()
            plt.scatter(usx, usy)
            plt.savefig('us_scatter/' + str(i))
            us = np.stack([usx, usy], axis=-1)
            model.place_cells.us.load(us, sess)
            print('box width: ' + str(np.round(box_width, 2)))
            print('box height: ' + str(np.round(box_height, 2)))

        # if (i + 1) % (10 * flags.save_interval) == 0:
        #     save_name = flags.run_ID
        #     visualize.save_autocorr(
        #         sess, model, save_name, step=(i + 1), flags=flags
        #     )



def main(argv):
    np.random.seed(1)

    # Set up checkpoint dirs
    ckpt_dir = flags.save_dir + "/ckpts"
    log_dir = flags.save_dir + "/logs"
    if not os.path.exists(flags.save_dir):
        os.mkdir(flags.save_dir)

    model = Model(flags)
    data_manager = DataManager(flags)
    trainer = Trainer(model, flags)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # For Tensorboard log
    log_dir = flags.save_dir + "/" + flags.run_ID + "/log"
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # Load checkpoints
    saver, start_step = load_checkpoints(sess)

    # Save params to file
    save_params(sess)

    # Train
    if flags.meta:
        meta_train(sess, model, trainer, saver, summary_writer, start_step, data_manager)
    else:
        train(sess, model, trainer, saver, summary_writer, start_step, data_manager)


if __name__ == '__main__':
    tf.app.run()
