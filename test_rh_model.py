# encoding=utf-8
"""
加载模型 并测试 image 文件 将结果保存为 h5
"""

import os
import sys
import numpy as np
import tensorflow as tf
import h5py
from read_sample import RoadHackersSampleReader

from retrain import IMAGE_DATA_TENSOR_NAME
from retrain import BOTTLENECK_TENSOR_NAME
from retrain import BottleneckCache
from retrain import run_bottleneck_on_image


def restore_model(sess):
    """
    restore model from file
    :param sess:
    :return:
    """
    model_meta_file = sys.argv[1]
    print "model_meta_file=%s" % model_meta_file
    model_dir = os.path.dirname(model_meta_file)
    print "model_dir=%s" % model_dir

    saver = tf.train.import_meta_graph(model_meta_file)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    graph = tf.get_default_graph()
    image_data_tensor = graph.get_tensor_by_name(IMAGE_DATA_TENSOR_NAME)
    bottleneck_tensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
    final_tensor = graph.get_tensor_by_name('final_result:0')
    return image_data_tensor, bottleneck_tensor, final_tensor


def save_h5_result(d, file_path):
    """
    save result into h5 file
    :param dict:
    :param file_path:
    :return:
    """
    # data=np.array(dict.items())
    data = np.array(d.items(), dtype=np.float64)
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('attrs', data=data)
    print("saved {0} bottlenecks to {1}".format(len(d), file_path))


if __name__ == "__main__":
    sess = tf.Session()

    # restore model and get tensors
    image_data_tensor, bottleneck_tensor, final_tensor = restore_model(sess)

    # read test features
    reader = RoadHackersSampleReader()
    reader.read_test_feature_files(sys.argv[2])

    result_dict = {}
    step = 100
    sample_sig = reader.get_sample_signature()
    bn_cache = BottleneckCache(sample_sig)
    s = 0
    bn_values = []
    time_list = []
    for vs, idxs in reader.test_image_next_batch(-1):
        s += 1
        if s % step == 0:
            print 'finish calc test image %d' % s
        for v, idx in zip(vs, idxs):
            time_list.append(v[1])
            image = v[2]
            bn_value = bn_cache.get(idx)
            if bn_value is None:
                bn_value = run_bottleneck_on_image(sess,
                                                   image, image_data_tensor,
                                                   bottleneck_tensor)
            bn_values.append(np.squeeze(np.array(bn_value)))

    final_tensor_value = sess.run(
        [final_tensor],
        feed_dict={bottleneck_tensor: bn_values}
    )
    final_predict = final_tensor_value[0] - 0.5
    for t, v in zip(time_list, final_predict):
        result_dict[t] = v
    save_h5_result(result_dict, sys.argv[3])

    # predict them
    print final_predict
