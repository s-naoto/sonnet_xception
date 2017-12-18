# coding: utf-8
import tensorflow as tf
import sonnet as snt
import numpy as np
from PIL import Image


def _one_hot(length, value):
    tmp = np.zeros(length, dtype=np.float32)
    tmp[value] = 1.
    return tmp


class FlowerDataSet(snt.AbstractModule):
    def __init__(self, file_list, image_size, batch, name='FlowerDataSet'):
        super(FlowerDataSet, self).__init__(name=name)
        self.filelist = file_list

        imgs = []
        lbls = []
        self.image_size = image_size
        for f in self.filelist:
            img = Image.open(f)
            img = img.resize(self.image_size)
            imgs.append(np.asarray(img))
            lbls.append(_one_hot(17, (int(f.split('_')[-1][:-4]) - 1) // 80))
        imgs = np.asarray(imgs)
        lbls = np.asarray(lbls)

        self.num_data = len(self.filelist)
        self.images = tf.constant(imgs)
        self.labels = tf.constant(lbls, dtype=tf.float32)

        self.batch = batch

    def _build(self, is_train):
        if is_train:
            indices = tf.random_uniform([self.batch], 0, self.num_data, tf.int64)
            x_ = tf.cast(tf.gather(self.images, indices), tf.float32)
            # flip left-right
            x_tmp = tf.split(x_, self.batch, axis=0)
            flipped = []
            for x_t in x_tmp:
                flipped.append(tf.image.random_flip_left_right(tf.squeeze(x_t)))
            distorted_x = tf.stack(flipped, axis=0)
            # brightness
            distorted_x = tf.image.random_brightness(distorted_x, max_delta=63)
            # contrast
            x = tf.image.random_contrast(distorted_x, lower=0.2, upper=1.8)

            y_ = tf.gather(self.labels, indices)

            return x, y_
        else:
            return tf.cast(self.images, tf.float32), self.labels

    @staticmethod
    def cost(logits, target):
        return tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=target)

    @staticmethod
    def evaluation(logits, target):
        correct_prediction = tf.equal(tf.argmax(logits, 1, name='argmax_y'), tf.argmax(target, 1, name='argmax_t'))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
