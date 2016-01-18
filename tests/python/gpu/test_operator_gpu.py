import os
import sys
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '..', 'unittest'))
print sys.path
from test_operator import *

if __name__ == '__main__':
    test_softmax_with_shape((3,4), mx.gpu())
    test_multi_softmax_with_shape((3,4,5), mx.gpu())
