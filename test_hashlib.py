"""
just testing
"""

import hashlib
import sys
import h5py
import time

if __name__ == '__main__':
    xObj = h5py.File(sys.argv[1], 'r')

    start = time.time()
    num = 0
    for t in xObj:
        num += 1
        img = xObj[t]
        md5 = hashlib.md5(img).hexdigest()
        print md5
    end = time.time()

    print 'calc %d cost %d seconds' % (num, end - start)
