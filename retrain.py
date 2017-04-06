# encoding=utf-8
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple transfer learning with an Inception v3 architecture model which
displays summaries in TensorBoard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import simplejson as json
from redis import StrictRedis

from tensorflow.python.platform import gfile

from read_sample import RoadHackersSampleReader

FLAGS = None

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # 要哪一层的产出 tensor
BOTTLENECK_TENSOR_SIZE = 2048
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 应该是第一层 input 层 直接接原始 JPEG raw data
IMAGE_DATA_TENSOR_NAME = 'DecodeJpeg:0'  # decode 之后的 图像内容数据
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'  #

def create_inception_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
      bottleneck_tensor
      jpeg_data_tensor: image data of raw jpeg
      resized_input_tensor: np arr values of image
    """
    with tf.Session() as sess:
        model_filename = os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb')  # 老模型pb 定义文件
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, image_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, IMAGE_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, image_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      bottleneck_tensor: Layer before the final softmax.

    Returns:
      Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)  # squeeze 的作用是将没用的1维压缩
    return bottleneck_values


def maybe_download_and_extract():
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    """
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            # sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                 filepath,
                                                 _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def get_np_hash(arr, isDataSet=False):
    """
    计算 np_array 的 hash 值
    :param np_array:
    :return:
    """
    if isDataSet:
        s = ''.join(repr(v) for v in arr)
    else:
        s = json.dumps(arr.tolist())
    return hashlib.md5(s).hexdigest()


def get_or_create_bottleneckcache(sess, data_reader, bn_cache, image_data_tensor, bottleneck_tensor):
    """
    对于所有样本的 bottleneck_value 如果能从cache 恢复就拿，没有就计算
    :param sess: active session
    :param data_reader: sample reader
    :param bn_cache: bottleneck cache instance
    """
    data_reader.use_all()  # 先获得全部样本 pre-train
    how_many_bottlenecks = 0
    for img_np_datas, labels, idxs in data_reader.next_batch(-1):
        for img_np_data, label, idx in zip(img_np_datas, labels, idxs):
            # img_hash_key = get_np_hash(img_np_data, isDataSet=True)
            bottleneck_value = bn_cache.get(idx)
            if bottleneck_value is None:
                print("need calc bottleneck: %s" % idx)
                # sys.stdout.flush()
                bottleneck_value = run_bottleneck_on_image(sess, img_np_data, image_data_tensor, bottleneck_tensor)
                bn_cache.set(idx, bottleneck_value)
            else:
                print("succ get bottleneck: %s" % idx)
                # sys.stdout.flush()
            how_many_bottlenecks += 1
            if how_many_bottlenecks % 1000 == 0:
                print("{0} bottlenecks calc done.".format(how_many_bottlenecks))
                # sys.stdout.flush()
                # bn_cache.save_to_file(FLAGS.bottleneck_file)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
    """Adds a new softmax and fully-connected layer for training.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

    Args:
      class_count: Integer of how many categories of things we're trying to
      recognize.
      final_tensor_name: Name string for the new final node that produces results.
      bottleneck_tensor: The output of the main CNN graph.

    Returns:
      The tensors for the training and cross entropy results, and tensors for the
      bottleneck input and ground truth input.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')  # 这种有 default 值的 ph 用这个 api

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    stddev = 0.001
    params = [
        ([BOTTLENECK_TENSOR_SIZE, 1024], [1024]),
        ([1024, 512], [512]),
        ([512, 256], [256]),
        ([256, 128], [128]),
        ([128, 1], [1])
    ]
    '''
    params = [
        ([BOTTLENECK_TENSOR_SIZE, 1], [1])
    ]
    '''
    W_list = []
    with tf.name_scope(layer_name):
        for i in range(len(params)):
            name = 'hw%d' % i
            Wp = params[i][0]
            with tf.name_scope(name):
                W = tf.Variable(tf.truncated_normal(Wp, stddev=stddev), name=name)
                W_list.append(W)
                variable_summaries(W)
            name = 'hb%d' % i
            bp = params[i][1]
            with tf.name_scope(name):
                b = tf.Variable(tf.zeros(bp), name=name)
                variable_summaries(b)
            name = 'Wx_b%d' % i
            with tf.name_scope(name):
                if i == 0:
                    logits = bottleneck_input
                if i != len(params) - 1:
                    logits = tf.nn.relu(tf.matmul(logits, W) + b)
                else:
                    logits = tf.nn.sigmoid(tf.matmul(logits, W) + b, name=final_tensor_name)  # 这里改成 sigmoid 0-1
    final_tensor = logits

    tf.summary.histogram('activations', final_tensor)  # final_tensor.shape = [None, class_count]

    # 对 ground_truth_input 做一个变换，因为本身 label 的值域是[-0.5, 0.5]，这里统一加上0.5 再去和 logits 比
    ground_truth_input_add = tf.add(ground_truth_input, 0.5)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=ground_truth_input_add)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            mse = calc_mse(logits, ground_truth_input_add)

    layer_name = "target_variables"
    with tf.name_scope(layer_name):
        tf.summary.scalar('mean_cross_entropy', cross_entropy_mean)
        tf.summary.scalar('mean_square_err', mse)

    l2_loss_rate = 0.001
    with tf.name_scope('train'):
        # optimizer = tf.train.GradientDescentOptimizer
        optimizer = tf.train.AdamOptimizer
        l2_loss = 0
        for w in W_list:
            l2_loss += l2_loss_rate * tf.nn.l2_loss(w)
        loss = cross_entropy_mean + l2_loss
        # loss = mse + l2_loss
        # loss = mse
        # loss = cross_entropy_mean
        train_step = optimizer(FLAGS.learning_rate).minimize(loss)
    return (train_step, mse, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def calc_mse(tensor1, tensor2):
    return tf.reduce_mean(tf.square(tensor1 - tensor2))


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data
      into.

    Returns:
      Tuple of (evaluation step, prediction).
    """
    ground_truth_tensor_add = tf.add(ground_truth_tensor, 0.5)
    mse = calc_mse(result_tensor, ground_truth_tensor_add)
    return mse, result_tensor


class BottleneckCache:
    """
    存储 Bottleneck 的结果
    利用外部 redis kv server 解决 kv 数据过大问题
    """

    def __init__(self, sample_signature):
        idx_db = StrictRedis(host='127.0.0.1',
                             port=6379,
                             db=0)

        max_idx = idx_db.get('max_idx')
        if not max_idx:
            idx_db.set('max_idx', 1)

        self.db_idx = idx_db.get(sample_signature)
        if self.db_idx is not None:
            print("succ find bottleneckcache by {} db={}".format(sample_signature, self.db_idx))
        else:
            for other_signature in idx_db.scan_iter():
                if other_signature.startswith(sample_signature) or sample_signature.startswith(other_signature):
                    self.db_idx = idx_db.get(other_signature)
                    print("succ near find bottleneckcache by "
                          "sample_sig {} and other_sig {} db={}".format(sample_signature, other_signature, self.db_idx))
                    break
        if self.db_idx is None:
            print("cannot find bottleneckcache by {}".format(sample_signature))
            self.db_idx = int(idx_db.get('max_idx')) + 1
            idx_db.set(sample_signature, self.db_idx)
            idx_db.set('max_idx', self.db_idx)

        self.rdb = StrictRedis(host='127.0.0.1',
                               port=6379,
                               db=self.db_idx)

    def flush(self):
        self.rdb.flushdb()

    def feed(self, d):
        for k, v in d.items():
            self.set(k, v)

    def set(self, k, v):
        self.rdb.set(k, json.dumps(v.tolist()))

    def get(self, key):
        v = self.rdb.get(key)
        if v:
            return np.array(json.loads(v))
        else:
            return v

    def size(self):
        return self.rdb.dbsize()

    '''
    def save_to_file(self, file_path):
      import h5py
      with h5py.File(file_path, 'w') as f:
        for k, v in self.bottleneck_map.items():
          f.create_dataset(str(k), data=np.array(v, dtype=np.float32))
      print("saved {0} bottlenecks to {1}".format(len(self.bottleneck_map), file_path))

    def load_from_file(self, file_path):
      import h5py
      with h5py.File(file_path, 'r') as f:
        for k in f:
          self.bottleneck_map[int(k)] = np.array(f[k][:])
      print("loaded {0} bottlenecks from {1}".format(len(self.bottleneck_map), file_path))
  '''


def get_baseline_mse(t, sess):
    num = len(t)
    mse = calc_mse(t, np.zeros([num, 1], dtype=np.float32))
    return mse.eval(session=sess)


def main(_):
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # Set up the pre-trained graph.
    maybe_download_and_extract()  # 拿到 Inception 的模型pb文件
    graph, bottleneck_tensor, image_data_tensor, resized_image_tensor = (
        create_inception_graph())  # load it

    # 读取样本
    data_reader = RoadHackersSampleReader()
    data_reader.read_sample_dir(FLAGS.sample_dir)
    sample_signature = data_reader.get_sample_signature()

    # Look at the folder structure, and create lists of all the images.
    # image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
    #                                  FLAGS.validation_percentage)

    # See if the command-line flags mean we're applying any distortions.
    '''
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)
    '''
    # 创建 sess
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))

    # bottleneck缓存到文件
    bn_cache = BottleneckCache(sample_signature)  # 样本文件的组合和 cache 是一一对应的. 避免混淆
    get_or_create_bottleneckcache(sess, data_reader, bn_cache, image_data_tensor, bottleneck_tensor)

    '''
    if do_distort_images:
      # We will be applying distortions, so setup the operations we'll need.
      distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(
          FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
          FLAGS.random_brightness)
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir,
                        jpeg_data_tensor, bottleneck_tensor)
    '''

    # Add the new layer that we'll be training.
    class_count = 1
    (train_step, new_mse, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(class_count,
                                            FLAGS.final_tensor_name,
                                            bottleneck_tensor)

    # Create the operations we need to evaluate the accuracy of our new layer.
    final_mse, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)  # 这是最后一步，session 的输出 变成 accuracy 和 prediction, 其他的都是中间tensor 了

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    data_reader.use_train()
    # Run the training for as many cycles as requested on the command line.
    batch_interval = 50
    batchs = 0
    for i in range(FLAGS.how_many_training_steps):  # training steps
        # Get a batch of input bottleneck values, either calculated fresh every time
        # with distortions applied, or from the cache stored on disk.
        for xs, train_ground_truth, idxs in data_reader.next_batch(FLAGS.train_batch_size):
            batchs += 1
            # print("It is batch:", batchs)
            # sys.stdout.flush()
            train_bottlenecks = [bn_cache.get(idx) for idx in idxs]

            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run([merged, train_step],
                                        feed_dict={bottleneck_input: train_bottlenecks,
                                                   ground_truth_input: train_ground_truth})  # 真正去 training
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step or batchs % batch_interval == 0:
                new_mse_value, cross_entropy_value = sess.run(
                    [final_mse, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                baseline_mse = get_baseline_mse(train_ground_truth, sess)
                # 也许这里再打出 training 的情况没必要? 直接打 validation 情况如何
                print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))
                print('%s: Step %d: Tbase mse = %f' % (datetime.now(), i, baseline_mse))
                print('%s: Step %d: Train mse = %f' % (datetime.now(), i, new_mse_value))
                sys.stdout.flush()
                '''
                validation_bottlenecks, validation_ground_truth, _ = (
                    get_random_cached_bottlenecks(
                        sess, image_lists, FLAGS.validation_batch_size, 'validation',
                        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                        bottleneck_tensor))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = sess.run(
                    [merged, accuracy],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (datetime.now(), i, validation_accuracy * 100,
                       len(validation_bottlenecks)))
               '''

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    data_reader.use_test()
    for xs, test_ground_truth, idxs in data_reader.next_batch(FLAGS.test_batch_size):
        test_bottlenecks = [bn_cache.get(idx) for idx in idxs]
        # test_bottlenecks = np.array(test_bottlenecks).reshape([-1, 2048])
        final_mse_value, predictions = sess.run(
            [final_mse, final_tensor],
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})
        test_sample_num = len(test_bottlenecks)
        baseline_mse = get_baseline_mse(test_ground_truth, sess)
        print('Baseline test mse = %f' % (baseline_mse))
        print('Final test mse = %f (N=%d)' % (final_mse_value, test_sample_num))
        sys.stdout.flush()

    # 保存模型到硬盘.
    saver = tf.train.Saver()
    save_path = saver.save(sess, FLAGS.final_model_path)
    print("Final Model saved in file: %s" % save_path)

    '''
    if FLAGS.print_misclassified_test_images:
      print('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          print('%70s  %s' % (test_filename, image_lists.keys()[predictions[i]]))
    '''

    # Write out the trained graph and labels with the weights stored as constants.
    '''
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
        # with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        #  f.write('\n'.join(image_lists.keys()) + '\n')
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_dir',
        type=str,
        default='',
        help='Path to folders of samples'
    )
    parser.add_argument(
        '--bottleneck_file',
        type=str,
        default='./bottleneck_cache.h5',
        help='File name of bottleneck cache.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='/tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='/tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=1,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=5,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="""\
      Whether to print out a list of all misclassified test images.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/imagenet',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--final_model_path',
        type=str,
        default='/tmp/rh_final_model',
        help='file to store final model we trained.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
0
