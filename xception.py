# coding: utf-8

import tensorflow as tf
import sonnet as snt


# entry flow module
class EntryFlowModule(snt.AbstractModule):
    def __init__(self, output_channels, num, name='EntryFlowModule'):
        self.output_channels = output_channels
        self.num = str(num)

        super(EntryFlowModule, self).__init__(name=name)

        with self._enter_variable_scope():
            self.resconv_e1 = snt.Conv2D(output_channels=self.output_channels, kernel_shape=1, stride=2, name='resconv_e{}'.format(self.num))
            self.bn_rese1 = snt.BatchNorm(name='bn_rese{}'.format(self.num))
            self.sepconv_e1 = snt.SeparableConv2D(output_channels=self.output_channels, channel_multiplier=1, kernel_shape=3, name='sepconv_e{}1'.format(self.num))
            self.bn_sepe1 = snt.BatchNorm(name='bn_sepe{}1'.format(self.num))
            self.sepconv_e2 = snt.SeparableConv2D(output_channels=self.output_channels, channel_multiplier=1, kernel_shape=3, name='sepconv_e{}2'.format(self.num))
            self.bn_sepe2 = snt.BatchNorm(name='bn_sepe{}2'.format(self.num))

    def _build(self, x, is_train):

        residual = self.resconv_e1(x)
        residual = self.bn_rese1(residual, is_train)
        h = self.sepconv_e1(x)
        h = self.bn_sepe1(h, is_train)
        h = self.sepconv_e2(tf.nn.relu(h, name='relu_e{}'.format(self.num)))
        h = self.bn_sepe2(h, is_train)
        h = tf.nn.max_pool(h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool_e{}'.format(self.num))
        h = tf.add(h, residual, name='add_e{}'.format(self.num))

        return h


# middle flow module
class MiddleFlowModule(snt.AbstractModule):
    def __init__(self, num, name='MiddleFlowModule'):
        self.num = str(num)
        super(MiddleFlowModule, self).__init__(name=name)

        with self._enter_variable_scope():
            self.sepconv_m1 = snt.SeparableConv2D(output_channels=728, channel_multiplier=1, kernel_shape=3, name='sepconv_m{}1'.format(self.num))
            self.bn_sepm1 = snt.BatchNorm(name='bn_sepm{}1'.format(self.num))
            self.sepconv_m2 = snt.SeparableConv2D(output_channels=728, channel_multiplier=1, kernel_shape=3, name='sepconv_m{}2'.format(self.num))
            self.bn_sepm2 = snt.BatchNorm(name='bn_sepm{}2'.format(self.num))
            self.sepconv_m3 = snt.SeparableConv2D(output_channels=728, channel_multiplier=1, kernel_shape=3, name='sepconv_m{}3'.format(self.num))
            self.bn_sepm3 = snt.BatchNorm(name='bn_sepm{}3'.format(self.num))

    def _build(self, x, is_train):
        res = x
        h = self.sepconv_m1(tf.nn.relu(x, name='relu_m{}1'.format(self.num)))
        h = self.bn_sepm1(h, is_train)
        h = self.sepconv_m2(tf.nn.relu(h, name='relu_m{}2'.format(self.num)))
        h = self.bn_sepm2(h, is_train)
        h = self.sepconv_m3(tf.nn.relu(h, name='relu_m{}3'.format(self.num)))
        h = self.bn_sepm3(h, is_train)
        h = tf.add(h, res, name='add_m{}'.format(self.num))

        return h


# exit flow module
class ExitFlowModule(snt.AbstractModule):
    def __init__(self, name='ExitFlowModule'):
        super(ExitFlowModule, self).__init__(name=name)

        with self._enter_variable_scope():
            self.resconv_ex1 = snt.Conv2D(output_channels=1024, kernel_shape=1, stride=2, name='resconv_ex1')
            self.bn_resex1 = snt.BatchNorm(name='bn_resex1')
            self.sepconv_ex1 = snt.SeparableConv2D(output_channels=728, channel_multiplier=1, kernel_shape=3, name='sepconv_ex1')
            self.bn_sepex1 = snt.BatchNorm(name='bn_sepex1')
            self.sepconv_ex2 = snt.SeparableConv2D(output_channels=1024, channel_multiplier=1, kernel_shape=3, name='sepconv_ex2')
            self.bn_sepex2 = snt.BatchNorm(name='bn_sepex2')
            self.sepconv_ex3 = snt.SeparableConv2D(output_channels=1536, channel_multiplier=1, kernel_shape=3, name='sepconv_ex3')
            self.bn_sepex3 = snt.BatchNorm(name='bn_sepex3')
            self.sepconv_ex4 = snt.SeparableConv2D(output_channels=2048, channel_multiplier=1, kernel_shape=3, name='sepconv_ex4')
            self.bn_sepex4 = snt.BatchNorm(name='bn_sepex4')

    def _build(self, x, is_train):
        residual = self.resconv_ex1(x)
        residual = self.bn_resex1(residual, is_train)
        h = self.sepconv_ex1(tf.nn.relu(x, name='relu_ex1'))
        h = self.bn_sepex1(h, is_train)
        h = self.sepconv_ex2(tf.nn.relu(h, name='relu_ex2'))
        h = self.bn_sepex2(h, is_train)
        h = tf.nn.max_pool(h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool_ex')
        h = tf.add(h, residual, name='add_ex2')
        h = self.sepconv_ex3(h)
        h = tf.nn.relu(self.bn_sepex3(h, is_train), name='relu_ex3')
        h = self.sepconv_ex4(h)
        h = tf.nn.relu(self.bn_sepex4(h, is_train), name='relu_ex4')
        # in paper, kernel size of global pooling is 10.
        # in this code, kernel size is 5 cause length of input image is 149.
        h = tf.nn.avg_pool(h, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID', name='global_avg_pool')
        h = snt.BatchFlatten(name='flatten')(h)

        return h


# Xception module
class Xception(snt.AbstractModule):
    def __init__(self, output_size, name='Xception'):

        self.output_size = output_size

        super(Xception, self).__init__(name=name)
        # input size = (149, 149)
        with self._enter_variable_scope():
            # Entry Flow
            self.conv_e1 = snt.Conv2D(output_channels=32, kernel_shape=3, stride=2, name='conv_e1')
            self.bn_e1 = snt.BatchNorm(name='bn_e1')
            self.conv_e2 = snt.Conv2D(output_channels=64, kernel_shape=3, name='conv_e2')
            self.bn_e2 = snt.BatchNorm(name='bn_e2')

            self.entry_flow_1 = EntryFlowModule(output_channels=128, num=1, name='entry_flow_1')
            self.entry_flow_2 = EntryFlowModule(output_channels=256, num=2, name='entry_flow_2')
            self.entry_flow_3 = EntryFlowModule(output_channels=728, num=3, name='entry_flow_3')

            # Middle Flow
            self.middles = [MiddleFlowModule(num=n, name='middle_flow_{}'.format(str(n))) for n in range(1, 9)]

            # Exit Flow
            self.exit_flow = ExitFlowModule(name='exit_flow')

            self.l1 = snt.Linear(output_size=256, name='l1')
            self.l2 = snt.Linear(output_size=self.output_size, name='l2')

    def _build(self, inputs, is_train, dropout_rate=0.5):
        # input size = (149, 149)
        # in paper, input image size = (299, 299)

        # Entry Flow
        h = self.conv_e1(inputs)
        h = tf.nn.relu(self.bn_e1(h, is_train), name='relu_1')
        h = self.conv_e2(h)
        h = tf.nn.relu(self.bn_e2(h, is_train), name='relu_2')

        h = self.entry_flow_1(h, is_train)
        h = self.entry_flow_2(h, is_train)
        h = self.entry_flow_3(h, is_train)

        # Middle Flow
        for mm in self.middles:
            h = mm(h, is_train)

        # Exit Flow
        h = self.exit_flow(h, is_train)

        h = tf.nn.relu(self.l1(h), name='relu_l1')
        feature = h
        h = tf.nn.dropout(h, keep_prob=dropout_rate if is_train else 1., name='dropout')
        y = self.l2(h)

        return y, feature
