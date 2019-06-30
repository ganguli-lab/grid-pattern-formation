# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

from model import Model
from trainer import Trainer
from options import get_options
import visualize
import pickle

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
    sess, model, trainer, saver, summary_writer, start_step
):
    for i in range(start_step, flags.steps):
        trainer.train(sess, summary_writer, step=i, flags=flags)

        if i % flags.save_interval == flags.save_interval - 1:
            save_checkponts(sess, saver, i)
            save_name = flags.run_ID
            visualize.save_visualization(
                sess, model, save_name, step=i, flags=flags
            )

        # if (i + 1) % (10 * flags.save_interval) == 0:
        #     save_name = flags.run_ID
        #     visualize.save_autocorr(
        #         sess, model, save_name, step=(i + 1), flags=flags
        #     )


def main(argv):
    np.random.seed(1)

    ckpt_dir = flags.save_dir + "/ckpts"
    log_dir = flags.save_dir + "/logs"
    if not os.path.exists(flags.save_dir):
        os.mkdir(flags.save_dir)

    model = Model(flags)
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
    train(sess, model, trainer, saver, summary_writer, start_step)


if __name__ == '__main__':
    tf.app.run()
