# encoding=utf-8
"""这是读 roadhackers 样本 读取代码的封装
"""

import h5py
import numpy as np
import random
from datetime import datetime
from bi_search import BiSearch
import os
import os.path
from exceptions import IOError


class RoadHackersSampleReader:
    def __init__(self):
        """
        """
        self.train_rate = 0
        self.valid_rate = 0

        self.iidxs = []
        self.img_lbels_arr = []
        self.start_iidx = 0
        self.end_iidx = 0

        self.sorted_sample_files_string = ''
        self.img_labels_arr = []
        pass

    def get_sample_size(self):
        """
        get sample size
        :return:
        """
        return len(self.img_labels_arr)

    def get_sample_signature(self):
        """
        由于每次训练可能采用不同的样本集合（不同的外部样本文件）
        而 bottleneck cache 文件可能会复用，为了避免混淆导致错乱，给每个样本集合制作一个签名
        在外部管理一个签名到 kv db 的 idx 的映射, 这样就能确保 cache 不混淆
        :return:
        """
        assert (len(self.sorted_sample_files_string) > 0)
        return self.sorted_sample_files_string

    def read_sample_dir(self, sample_dir):
        """
        从样本 dir 中分别读取 feature_dir 和 label_dir 的文件
        :param sample_dir:
        :return:
        """
        if not os.path.exists(sample_dir):
            raise IOError('dir does not exist: %s' % sample_dir)

        xfiles, yfiles = self.walk_sample_dir(sample_dir + '/features', sample_dir + '/labels')
        assert (len(xfiles) == len(yfiles))
        assert (len(xfiles) > 0)
        self.read_sample_files(xfiles, yfiles)

    def walk_sample_dir(self, feature_dir, label_dir):
        """
        遍历样本目录，找到feature 和 label 文件
        :param
        :return: [xfiles, yfiles] pair list
        """
        xfiles = []
        yfiles = []
        for root, dirs, files in os.walk(feature_dir):
            files = filter(lambda x: x.endswith('.h5'), files)
            files.sort()
            self.sorted_sample_files_string = ' '.join(files)
            for f in files:  # 这里一定是顺序的 很重要！ 因为涉及到 bottleneck 的 cache
                basename = os.path.basename(f)
                feature_file = feature_dir + '/' + basename
                label_file = label_dir + '/' + basename
                if os.path.exists(feature_file) and os.path.exists(label_file):
                    xfiles.append(feature_file)
                    yfiles.append(label_file)
                else:
                    print("can't find feature file "
                          "{0} or label file: {1}".format(feature_file, label_file))
        return xfiles, yfiles

    def read_test_feature_files(self, feature_dir):
        """
        predict时候，只读取 feature 文件
        :param xfiles:
        :return:
        """
        self.test_image_arr = []
        for root, dirs, files in os.walk(feature_dir):
            files = filter(lambda x: x.endswith('.h5'), files)
            files.sort()
            self.sorted_sample_files_string = ' '.join(files)
            for fidx, fx in enumerate(files):
                basename = os.path.basename(fx)
                feature_file = feature_dir + '/' + basename
                xObj = h5py.File(feature_file, 'r')
                for t in xObj:
                    image = xObj[t]
                    self.test_image_arr.append((fidx, t, image))

    def test_image_next_batch(self, num):
        """
        返回 next_batch features
        :param num: -1 全部返回
        :return:
        """
        if num <= 0:
            yield self.test_image_arr, range(0, len(self.test_image_arr))
        else:
            for j in range(0, len(self.test_image_arr) / num):
                yield self.test_image_arr[j * num:(j + 1) * num], range(j * num, (j + 1) * num)

    def read_sample_files(self, xfiles, yfiles):
        """
          :param xfiles: feature file list
          :param yfiles: label file list
          """
        self.img_labels_arr = []
        for fidx, (fx, fy) in enumerate(zip(xfiles, yfiles)):  # merge features and labels
            xObj = h5py.File(fx, 'r')
            yObj = h5py.File(fy, 'r')
            lables_arr = [v for v in yObj['attrs']]

            for t in xObj:
                img = xObj[t]
                vidx = BiSearch.search(lables_arr, t) - 1
                v = map(lambda x: float(x), lables_arr[vidx])
                self.img_labels_arr.append((fidx, t, img, v))
                """
                [fidx, utctime, [img], [lables]]
                img.shape = (320, 320, 3)
                lables = [time0, speed1, speed2, curv_time0, curv_time0+0.125, curv_time0+0.25, ... curv_time0+0.875]
                """
        print 'Read all file records finished. raw total record:', len(self.img_labels_arr)

        self.clean()
        self.shuffle()

    def clean(self):
        """
        对脏数据进行过滤: 转弯半径小于2m 的, 即曲率>0.5的, 这部分在比赛时亦不计入有效样本
        :return:
        """
        self.img_labels_arr = filter(lambda v: abs(float(v[3][3])) <= 0.5, self.img_labels_arr)

    def shuffle(self, train_rate=0.8, valid_rate=0.1):
        """
        can shuffle anytime if you want, 每次 shuffle 变换 train valid test 集合的内容
        :param train_rate:  训练集的比例
        :param valid_rate:  验证集的比例
                            1 - train_rate - valid_rate 是测试集的比例
        :return:
        """
        self.train_rate = train_rate
        self.valid_rate = valid_rate

        self.iidxs = range(0, len(self.img_labels_arr))

        # 分训练集 验证集 测试集
        np.random.shuffle(self.iidxs)  # shuffle 一次
        self.num_total = self.get_sample_size()
        self.num_train = int(self.num_total * self.train_rate)
        self.num_valid = int(self.num_total * self.valid_rate)
        self.num_test = self.num_total - self.num_train - self.num_valid

    def use_train(self):
        """
        训练集
        :return:
        """
        self.start_iidx = 0
        self.end_iidx = self.num_train
        print "use train. start_iidx=%d, end_iidx=%d" % (self.start_iidx, self.end_iidx)

    def use_valid(self):
        """
        验证集
        :return:
        """
        self.start_iidx = self.num_train
        self.end_iidx = self.num_train + self.num_valid
        print "use valid. start_iidx=%d, end_iidx=%d" % (self.start_iidx, self.end_iidx)

    def use_test(self):
        """
        测试集
        :return:
        """
        self.start_iidx = self.num_train + self.num_valid
        self.end_iidx = self.num_total
        print "use test. start_iidx=%d, end_iidx=%d" % (self.start_iidx, self.end_iidx)

    def use_all(self):
        """
        全集
        :return:
        """
        self.start_iidx = 0
        self.end_iidx = self.num_total
        print "use all. start_iidx=%d, end_iidx=%d" % (self.start_iidx, self.end_iidx)

    def pack_samples(self, iidxs, with_sidx=True):
        """
        将 samples 打包成需要的格式 返回
        :param iidxs:
        :return:
        """
        xs = []
        ys = []
        sidxs = []
        for i in iidxs:
            sidx = self.iidxs[i]
            sidxs.append(sidx)
            xs.append(self.img_labels_arr[sidx][2])
            ys.append(self.img_labels_arr[sidx][3][3])
        if with_sidx:
            return xs, np.reshape(ys, [-1, 1]), sidxs
        else:
            return xs, np.reshape(ys, [-1, 1])

    def next_batch(self, num, rand=True, with_sidx=True):
        """
        train.... valid... test..
        :param num: batch num, return all if num <= 0
        :param rand: rand如果是True 则每次随机在当前集合中取 否则顺序取
        :return:
        """
        if num <= 0:  # return all
            num = self.end_iidx - self.start_iidx
        if rand:
            iidxs = random.sample(self.iidxs[self.start_iidx:self.end_iidx], num)
            yield self.pack_samples(iidxs)
        else:
            # raise Exception("Not well implemented yet!!!") # 这里还没实现好  目前是调用1次 next_batch 就返回了所有样本  还没改好 不建议使用
            for j in range(0, (self.end_iidx - self.start_iidx) / num):  # 每次取 num个 取j轮
                iidxs = self.iidxs[j * num:(j + 1) * num]
                yield self.pack_samples(iidxs, with_sidx)

    def _get_str_time(self, timestamp):
        """
        获得字符串形式时间
        :param timestamp:
        :return:
        """
        timestamp = float(timestamp)
        return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')

    def show_sample_info(self, num, image_only=False):
        """
        用 pyplot 显示样本图片和label 信息
        :param num:
        :return:
        """
        import matplotlib.pyplot as plt
        max_sample_num = 6
        if num > max_sample_num:
            num = max_sample_num
        plt.title('sample info')
        fig = plt.figure()
        if image_only:
            for idx, sample in enumerate(random.sample(self.test_image_arr, num)):
                pos = 231 + idx
                ax = fig.add_subplot(pos)  # 3行2列
                ax.imshow(sample[2])
                fidx = str(sample[0])
                timeStrKey = self._get_str_time(sample[1])
                stri = 'File[' + fidx + '] ' + timeStrKey
                plt.xlabel(stri)
        else:
            for idx, sample in enumerate(random.sample(self.img_labels_arr, num)):  # 随机取 num 个 sample
                pos = 231 + idx
                ax = fig.add_subplot(pos)  # 3行2列
                ax.imshow(sample[2])
                fidx = str(sample[0])
                timeStrKey = self._get_str_time(sample[1])
                stri = 'File[' + fidx + '] ' + timeStrKey + ' \n ' + str(sample[3][3])
                plt.xlabel(stri)
        plt.show()

if __name__ == '__main__':
    data_reader = RoadHackersSampleReader()
    data_reader.read_sample_dir('./data/training')
    # data_reader.show_sample_info(20)
    data_reader.use_test()
    i = 0
    for i in range(5):
        print 'i:', i
        for a, b, idxs in data_reader.next_batch(-1, rand=True):
            print b
            print idxs
        if i > 3: break
        i += 1
