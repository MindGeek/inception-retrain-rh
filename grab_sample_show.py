"""
show sample image
"""
#encoding=utf-8

import sys
from read_sample import RoadHackersSampleReader

if __name__ == '__main__':
    data_reader = RoadHackersSampleReader()
    data_reader.read_sample_dir(sys.argv[1])
    data_reader.show_sample_info(10)

