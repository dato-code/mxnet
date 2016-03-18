import mxnet
import array
import unittest
import numpy as np

__has_sframe__ = False
__has_graphlab__ = False
try:
    import sframe as gl
    __has_sframe__ = True
except:
    pass

try:
    import graphlab as gl
    __has_graphlab__ = True
except:
    pass


@unittest.skipIf(__has_sframe__ is False and __has_graphlab__ is False, 'graphlab or sframe not found')
class SFrameIteratorBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if __has_sframe__ is False and __has_graphlab__ is False:
            return
        cls.data = gl.SFrame({'x': np.random.randn(10),
                              'y': np.random.randint(2, size=10)})
        cls.shape = [1]
        cls.label_field = 'y'
        cls.data_field = 'x'
        cls.data_size = len(cls.data)
        cls.data_expected = list(cls.data['x'])
        cls.label_expected = list(cls.data['y'])
        return cls

    def test_one_batch(self):
        it = mxnet.io.SFrameIter(self.data, data_field=self.data_field,
                                 data_shape=self.shape,
                                 label_field=self.label_field,
                                 batch_size=self.data_size)
        label_actual = []
        data_actual = []
        for d in it:
            data_actual.extend(d.data[0].asnumpy().flatten())
            label_actual.extend(d.label[0].asnumpy().flatten())
        np.testing.assert_almost_equal(label_actual, self.label_expected)
        np.testing.assert_almost_equal(data_actual, self.data_expected)

    def test_non_divisible_batch(self):
        batch_size = self.data_size + 1
        it = mxnet.io.SFrameIter(self.data, data_field=self.data_field,
                                 data_shape=self.shape,
                                 label_field=self.label_field,
                                 batch_size=batch_size)
        label_actual = []
        data_actual = []
        for d in it:
            data_actual.extend(d.data[0].asnumpy().flatten())
            label_actual.extend(d.label[0].asnumpy().flatten())

        # Truncate the batch to expected size, the truncated batch may contain arbitrary data.
        label_actual = label_actual[:len(self.label_expected)]
        data_actual = data_actual[:len(self.data_expected)]

        np.testing.assert_almost_equal(label_actual, self.label_expected)
        np.testing.assert_almost_equal(data_actual, self.data_expected)

    def test_padding(self):
        padding = 5
        batch_size = self.data_size + padding
        it = mxnet.io.SFrameIter(self.data, data_field=self.data_field,
                                 data_shape=self.shape,
                                 label_field=self.label_field,
                                 batch_size=batch_size)
        label_expected = self.label_expected + [0.0] * padding
        data_expected = self.data_expected + list(np.ndarray(self.shape).flatten()) * padding
        label_actual = []
        data_actual = []
        for d in it:
            data_actual.extend(d.data[0].asnumpy().flatten())
            label_actual.extend(d.label[0].asnumpy().flatten())
        np.testing.assert_almost_equal(label_actual, label_expected)
        np.testing.assert_almost_equal(data_actual, data_expected)

    def test_shape_inference(self):
        # TODO
        pass

class SFrameArrayIteratorTest(SFrameIteratorBaseTest):
    @classmethod
    def setUpClass(cls):
        if __has_sframe__ is False and __has_graphlab__ is False:
            return
        cls.data = gl.SFrame({'x': [np.random.randn(5)] * 10,
                              'y': np.random.randint(2, size=10)})
        cls.shape = [5]
        cls.label_field = 'y'
        cls.data_field = 'x'
        cls.data_size = len(cls.data)
        cls.data_expected = list(x for arr in cls.data['x'] for x in arr)
        cls.label_expected = list(cls.data['y'])
        return cls


class SFrameImageIteratorTest(SFrameIteratorBaseTest):
    @classmethod
    def setUpClass(cls):
        if __has_sframe__ is False and __has_graphlab__ is False:
            return
        w = 2
        h = 3
        c = 1
        d = w * h * c
        cls.data = gl.SFrame({'arr': [array.array('d', range(x, x+d)) for x in range(10)],
                              'y': np.random.randint(2, size=10)})
        cls.data['img'] = cls.data['arr'].pixel_array_to_image(w, h, c)
        cls.shape = (c, h, w)
        cls.label_field = 'y'
        cls.data_field = 'img'
        cls.data_size = len(cls.data)
        cls.data_expected = list(x for arr in cls.data['arr'] for x in arr)
        cls.label_expected = list(cls.data['y'])
        return cls


class SFrameMultiColumnIteratorTest(SFrameIteratorBaseTest):
    @classmethod
    def setUpClass(cls):
        # TODO
        pass
