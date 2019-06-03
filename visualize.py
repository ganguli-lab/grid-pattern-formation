# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import os
from scipy.misc import imsave
from scipy.signal import correlate2d
import cv2

import scores
import utils


def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret

def convert_to_colomap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def save_visualization(sess, model, save_name, step, flags):
    batch_size = flags.batch_size
    sequence_length = flags.sequence_length
    resolution = 20
    maze_extents = 1.1

    activations = np.zeros([flags.num_g_cells, resolution, resolution], dtype=np.float32) # (512, 32, 32)
    counts  = np.zeros([resolution, resolution], dtype=np.int32)        # (32, 32)

    index_size = 100

    for index in range(index_size):
        g, place_pos_batch = sess.run([model.g, model.target_pos])
        place_pos_batch = np.reshape(place_pos_batch, [-1, 2])

        for i in range(batch_size * sequence_length):
            pos_x = place_pos_batch[i,0]
            pos_z = place_pos_batch[i,1]
            x = (pos_x + maze_extents) / (maze_extents * 2) * resolution
            z = (pos_z + maze_extents) / (maze_extents * 2) * resolution
            if x >=0 and x < resolution and z >=0 and z < resolution:
                counts[int(x), int(z)] += 1
                activations[:, int(x), int(z)] += g[i, :]

    for x in range(resolution):
        for y in range(resolution):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]

    hidden_size = flags.num_g_cells

    cmap = matplotlib.cm.get_cmap('jet')

    images = []

    for i in range(hidden_size):
        im = activations[i,:,:]
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        im = convert_to_colomap(im, cmap)

        im = cv2.resize(im,
                        dsize=(resolution*2, resolution*2),
                        interpolation=cv2.INTER_NEAREST)
        # (40, 40, 4), uint8

        images.append(im)

    concated_image = concat_images_in_rows(images, 32, resolution*2)
    imdir = "images/" + save_name
    if not os.path.exists(imdir):
        os.mkdir(imdir)
    imsave(imdir + "/" + str(step+1) + ".png", concated_image)



def save_autocorr(sess, model, save_name, step, flags):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    coord_range=((-1.1, 1.1), (-1.1, 1.1))
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, coord_range, masks_parameters)
    
    res = dict()
    index_size = 100
    for _ in range(index_size):
      mb_res = sess.run({
          'pos_xy': model.target_pos,
          'bottleneck': model.g,
      })
      res = utils.concat_dict(res, mb_res)
        
    filename = save_name + '/autocorrs_' + str(step) + '.pdf'
    out = utils.get_scores_and_plot(
                latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                'images/', filename)
