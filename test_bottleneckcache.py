import unittest
import retrain
import numpy as np


class test_BottleneckCache(unittest.TestCase):
    def test_save_and_load(self):
        """
        testing
        :return:
        """
        file_path = './test_temp_file.h5'
        d0 = {0: np.array([.1, .2, .3]), 1: np.array([.4, .5, .6])}
        d1 = {0: np.array([.2, .3, .4]), 1: np.array([.5, .6, .7]), 2: np.array([.7])}

        bn_cache0 = retrain.BottleneckCache('file1 file2')
        bn_cache0.feed(d0)
        bn_cache1 = retrain.BottleneckCache('file1 file2 file3')
        bn_cache1.feed(d1)
        bn_cache02 = retrain.BottleneckCache('file1 file2')
        bn_cache11 = retrain.BottleneckCache('file1')

        self.assertEqual(bn_cache0.size(), 2)
        self.assertEqual(bn_cache1.size(), 3)
        self.assertTrue(np.allclose(d0[0], bn_cache0.get(0)))
        self.assertTrue(np.allclose(d1[1], bn_cache1.get(1)))
        self.assertEqual(bn_cache02.size(), 2)
        self.assertTrue(np.allclose(d0[1], bn_cache02.get(1)))
        self.assertEqual(bn_cache11.size(), 2)

if __name__ == '__main__':
    unittest.main()
