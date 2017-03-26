"""
jush testing
"""
import unittest
from read_sample import RoadHackersSampleReader

class TestRoadHackersSampleReader(unittest.TestCase):
    """
    just testing
    """
    def test_read_sample_dir(self):
        """
        to extract test code from __main__ to this file
        :return:
        """
        # TODO
        pass

    def test_read_test_feature_files(self):
        """
        testing
        :return:
        """
        data_reader = RoadHackersSampleReader()
        data_reader.read_test_feature_files(['./data/testing/657.h5'])
        count = 0
        for g in data_reader.test_image_next_batch(3):
            print 'one batch'
            for v in g:
                print v[1]
                print v[2]
                count += 1
        print count


if __name__ == '__main__':
    unittest.main()
