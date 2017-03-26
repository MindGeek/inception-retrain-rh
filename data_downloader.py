# encoding=utf-8

import os
import sys
import os.path
import urllib
from zipfile import ZipFile
from threading import Thread
from Queue import Queue


class RoadHackersDataDownloader(object):
    """
    封装样本下载 解压等耗时的工作 多线程
    """

    def __init__(self, feature_dir, label_dir,
                 num_download_worker=10, num_unzip_worker=10):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        assert (self.feature_dir != self.label_dir)  # avoid possible overwrite

        self.download_queue = Queue()
        self.unzip_queue = Queue()

        self.num_download_worker = num_download_worker
        self.num_unzip_worker = num_unzip_worker

        self.download_threads = []
        self.unzip_threads = []

        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        pass

    def thread_worker(self, job_queue, func):
        """each thread run this
        """
        while not job_queue.empty():
            job = job_queue.get()
            func(job)

    def download_till_finish(self):
        """
        启动多线程下载文件
        :return:
        """
        for i in range(self.num_download_worker):
            thread = Thread(target=self.thread_worker,
                            args=(self.download_queue,
                                  self.download_url))
            thread.start()
            self.download_threads.append(thread)

        for t in self.download_threads:
            t.join()

    def unzip_till_finish(self):
        """
        启动多线程解压文件
        :return:
        """
        for i in range(self.num_unzip_worker):
            thread = Thread(target=self.thread_worker,
                            args=(self.unzip_queue,
                                  self.unzip_file))
            thread.start()
            self.unzip_threads.append(thread)

        for t in self.unzip_threads:
            t.join()

    def download_url(self, job):
        """
        具体执行下载
        :param job:
        :return:
        """
        url, file = job
        if os.path.exists(file):  # file already exists
            print "%s already exists." % (file)
        print "downloading %s ..." % (url)
        ret = urllib.urlretrieve(url, file)
        if ret: print "download %s done." % (file)
        self.unzip_queue.put(file)  # 添加进解压队列

    def walk_and_download(self, feature_url_base, label_url_base, postfix,
                          start_idx, end_idx, flag="no_rewrite"):
        """
         遍历样本 url，并下载. 可以指定是否 rewrite 本地 file
        :param start_idx:
        :param end_idx:
        :param flag: "rewrite" or not
        """
        for i in range(start_idx, end_idx):
            # down feature
            url = "%s/%d%s" % (feature_url_base, i, postfix)
            file = "%s/%d%s" % (self.feature_dir, i, postfix)
            self.download_queue.put((url, file))

            # down label
            url = "%s/%d%s" % (label_url_base, i, postfix)
            file = "%s/%d%s" % (self.label_dir, i, postfix)
            self.download_queue.put((url, file))

        self.download_till_finish()
        # self.unzip_till_finish()

    def unzip_file(self, file_name):
        """
        解压文件并删除.zip
        :param file_name:
        :return:
        """
        dir_name = os.path.dirname(file_name)
        fsize = os.stat(file_name)
        if fsize < 1024:  # not big enough
            print "invalid zip file %s" % file_name
            return
        zipf = ZipFile(file_name, 'r')
        print "unziping %s ..." % file_name
        zipf.extractall(dir_name)
        zipf.close()
        print "here should rm %s" % file_name
        # os.remove(file_name)
        pass


if __name__ == "__main__":
    data_downloader = RoadHackersDataDownloader(sys.argv[1], sys.argv[2])
    data_downloader.walk_and_download('http://roadhack-sources.cdn.bcebos.com/train/image/',
                                      'http://roadhack-sources.cdn.bcebos.com/train/attr/',
                                      '.zip', 147, 150)
