# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import os
import datetime
from PIL import Image

from xception import Xception
from flower_dataset import FlowerDataSet


np.random.seed(4225)

# 利用GPUの設定
config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="1", allow_growth=True))

# パラメータをセット
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("num_training_iterations", 2000, "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100, "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_integer("batch_size", 100, "Batch size for training.")
tf.flags.DEFINE_integer("output_size", 17, "Size of output layer.")
tf.flags.DEFINE_float("weight_decay_rate", 1.e-4, "Rate for Weight Decay.")
tf.flags.DEFINE_float("train_ratio", 0.8, "Ratio of train data in the all data.")
tf.flags.DEFINE_float("init_lr", 1.e-3, "Initial learning rate.")
tf.flags.DEFINE_integer("decay_interval", 500, "lr decay interval.")
tf.flags.DEFINE_float("decay_rate", 0.5, "lr decay rate.")

# Xceptionモジュール
xception = Xception(FLAGS.output_size, name='Xception')

image_dir_path = 'jpg'
filelist = list(filter(lambda z: z[-4:] == '.jpg', os.listdir(image_dir_path)))
filelist = np.asarray([os.path.join(image_dir_path, f) for f in filelist])
idx = np.arange(len(filelist))
np.random.shuffle(idx)
border = int(len(filelist) * FLAGS.train_ratio)

# Flower data set モジュール
dataset_train = FlowerDataSet(file_list=filelist[:border], image_size=(149, 149), batch=FLAGS.batch_size, name='Flower_dataset_train')
dataset_test = FlowerDataSet(file_list=filelist[border:], image_size=(149, 149), batch=FLAGS.batch_size, name='Flower_dataset_test')

# 計算グラフ
# train
train_x, train_y_ = dataset_train(is_train=True)
train_y, _ = xception(train_x, is_train=True)
ce_loss = dataset_train.cost(logits=train_y, target=train_y_)
# L2正則化
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# total loss (with Weight Decay)
loss = tf.add(ce_loss, FLAGS.weight_decay_rate * tf.reduce_sum(reg_loss), name='total_loss')
tf.summary.scalar('loss', loss)
tf.summary.image('train_batch', train_x)

# test
test_x, test_y_ = dataset_test(is_train=False)
test_y, feature = xception(test_x, is_train=False)
accuracy = dataset_test.evaluation(logits=test_y, target=test_y_)
tf.summary.scalar('accuracy', accuracy)

# optimizer
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.train.exponential_decay(FLAGS.init_lr, global_step, FLAGS.decay_interval, FLAGS.decay_rate,
                                           staircase=True, name='learning_rate')
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
tf.summary.scalar('lr', learning_rate)


if __name__ == '__main__':
    # loggingレベルを設定
    tf.logging.set_verbosity(tf.logging.INFO)

    # 保存先ディレクトリ
    if not os.path.exists('summary'):
        os.mkdir('summary')
    log_dir = 'summary/' + datetime.datetime.strftime(datetime.datetime.today(), '%Y_%m_%d_%H_%M_%S')
    # summaryを統合
    merged = tf.summary.merge_all()

    test_labels = None
    with tf.Session(config=config) as sess:
        # パラメータを初期化
        sess.run(tf.global_variables_initializer())
        # writerをセット
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        # メインループ
        for training_iteration in range(1, FLAGS.num_training_iterations + 1):
            summary, train_loss_v, _ = sess.run((merged, loss, train_step))
            writer.add_summary(summary, training_iteration)

            if training_iteration % FLAGS.report_interval == 0:
                tf.logging.info("%d: Training loss %f.", training_iteration, train_loss_v)
        # 評価
        test_accuracy_v, test_feature, test_targets = sess.run((accuracy, feature, test_y_))
        test_labels = np.argmax(test_targets, axis=1)
        tf.logging.info("Test loss %f", test_accuracy_v)

        # featureをembedding_varとして保存
        embedding_var = tf.Variable(test_feature, trainable=False, name='embedding_variable')
        sess.run(tf.variables_initializer([embedding_var]))
        # モデルを保存
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(log_dir, "xception_flower17.ckpt"))

        # projectorの設定
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        # メタデータのファイルパス
        embedding.metadata_path = 'flower17_labels.tsv'
        # sprite画像のファイルパス
        embedding.sprite.image_path = 'flower17_sprite.jpg'
        # sprite画像の画像サイズ
        embedding.sprite.single_image_dim.extend([64, 64])
        projector.visualize_embeddings(writer, config)

    # メタデータ（ラベル情報）
    with open(os.path.join(log_dir, 'flower17_labels.tsv'), 'w') as f:
        f.write('file_name\tlabels\n')
        for im, l in zip(dataset_test.filelist, test_labels):
            f.write('{}\t{}\n'.format(im, l))

    # sprite画像
    images = [Image.open(f).resize((64, 64)) for f in dataset_test.filelist]
    rows = (int(np.sqrt(len(images))) + 1)
    master_width = master_height = rows * 64

    master = Image.new(
        mode='RGB',
        size=(master_width, master_height),
        color=(0, 0, 0))

    for n, img in enumerate(images):
        i = n % rows
        j = n // rows
        master.paste(img, (i * 64, j * 64))

    master.save(os.path.join(log_dir, 'flower17_sprite.jpg'))
