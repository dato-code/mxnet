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
    def setUp(self):
        self.data = gl.SFrame({'x': np.random.randn(10),
                              'y': np.random.randint(2, size=10)})
        self.shape = (1,)
        self.label_field = 'y'
        self.data_field = 'x'
        self.data_size = len(self.data)
        self.data_expected = list(self.data['x'])
        self.label_expected = list(self.data['y'])

    def test_one_batch(self):
        it = mxnet.io.SFrameIter(self.data, data_field=self.data_field,
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
        shape_total = 1
        for s in self.shape:
            shape_total *= s
        it = mxnet.io.SFrameIter(self.data, data_field=self.data_field,
                                 label_field=self.label_field,
                                 batch_size=batch_size)
        label_expected = self.label_expected + self.label_expected[:padding]
        data_expected = self.data_expected + self.data_expected[:(padding * shape_total)]
        label_actual = []
        data_actual = []
        for d in it:
            data_actual.extend(d.data[0].asnumpy().flatten())
            label_actual.extend(d.label[0].asnumpy().flatten())
        self.assertEqual(d.pad, padding)
        np.testing.assert_almost_equal(label_actual, label_expected)
        np.testing.assert_almost_equal(data_actual, data_expected)

    def test_shape_inference(self):
        it = mxnet.io.SFrameIter(self.data, data_field=self.data_field,
                                 label_field=self.label_field,
                                 batch_size=1)
        self.assertEquals(it.infer_shape()["final_shape"], self.shape)

    def test_missing_value(self):
        data = self.data.copy()
        if not isinstance(self.data_field, list):
            self.data_field = [self.data_field]
        for col in self.data_field:
            ls = list(data[col])
            ls[1] = None
            data[col] = ls
        it = mxnet.io.SFrameIter(data, data_field=self.data_field)
        self.assertRaises(lambda: [it])


class SFrameArrayIteratorTest(SFrameIteratorBaseTest):
    def setUp(self):
        self.data = gl.SFrame({'x': [np.random.randn(8)] * 10,
                              'y': np.random.randint(2, size=10)})
        self.shape = (8,)
        self.label_field = 'y'
        self.data_field = 'x'
        self.data_size = len(self.data)
        self.data_expected = list(x for arr in self.data['x'] for x in arr)
        self.label_expected = list(self.data['y'])

    def test_size1_array(self):
        # setup data
        self.data = gl.SFrame({'x': [np.random.randn(1)] * 10,
                              'y': np.random.randint(2, size=10)})
        self.shape = (1,)
        self.label_field = 'y'
        self.data_field = 'x'
        self.data_size = len(self.data)
        self.data_expected = list(x for arr in self.data['x'] for x in arr)
        self.label_expected = list(self.data['y'])

        self.test_one_batch()
        self.test_non_divisible_batch()
        self.test_padding()
        self.test_shape_inference()

    def test_zero_size_array(self):
        self.data = gl.SFrame()
        self.data['x'] = [array.array('d')] * 10
        it = mxnet.io.SFrameIter(self.data, data_field='x')
        data_actual = []
        for d in it:
            data_actual.extend(d.data[0].asnumpy().flatten())
        self.assertEquals(data_actual, [])

    def test_variable_size_array(self):
        self.data = gl.SFrame({'x': [[0], [0, 1], [0, 1, 2]]})
        self.assertRaises(ValueError, lambda: mxnet.io.SFrameIter(self.data, data_field='x'))


class SFrameImageIteratorTest(SFrameIteratorBaseTest):
    def setUp(self):
        w = 2
        h = 3
        c = 1
        d = 6
        n = 5
        self.data = gl.SFrame({'arr': [array.array('d', range(0, 6)),
                                       array.array('d', range(50, 56)),
                                       array.array('d', range(100, 106)),
                                       array.array('d', range(200, 206)),
                                       array.array('d', range(249, 255))],
                              'y': np.random.randint(2, size=n)})
        self.data['img'] = self.data['arr'].pixel_array_to_image(w, h, c)
        self.shape = (c, h, w)
        self.label_field = 'y'
        self.data_field = 'img'
        self.data_size = len(self.data)
        self.data_expected = list(x for arr in self.data['arr'] for x in arr)
        self.label_expected = list(self.data['y'])

    def test_encoded_image(self):
        # resize encodes the image
        self.data['img'] = gl.image_analysis.resize(self.data['img'], 2, 3, 1)
        self.test_shape_inference()
        self.test_padding()
        self.test_one_batch()
        self.test_missing_value()
        self.test_non_divisible_batch()

    def test_variable_size_image(self):
        shape1 = (2, 3, 1)
        shape2 = (2, 2, 2)
        tmp1 = gl.SArray([array.array('d', [0] * 6)])
        tmp2 = gl.SArray([array.array('d', [0] * 8)])
        data = gl.SFrame({'x': [tmp1.pixel_array_to_image(*shape1)[0], tmp2.pixel_array_to_image(*shape2)[0]]})
        it = mxnet.io.SFrameIter(data, data_field='x')
        self.assertRaises(lambda: [it])


class SFrameMultiColumnIteratorTest(SFrameIteratorBaseTest):
    def setUp(self):
        self.data = gl.SFrame({'i': [x for x in range(10)],
                              '-i': [-x for x in range(10)],
                              'f': [float(x) for x in range(10)],
                              '-f': [-float(x) for x in range(10)],
                              'arr': [range(2) for x in range(10)],
                              'y': np.random.randint(2, size=10)})
        self.shape = (6,)
        self.label_field = 'y'
        self.data_field = ['i', '-i', 'f', '-f', 'arr']
        self.data_size = len(self.data)
        def val_iter():
            for row in self.data:
                for col in self.data_field:
                    v = row[col]
                    if type(v) is array.array:
                        for x in v:
                            yield x
                    else:
                        yield float(v)
        self.data_expected = list(val_iter())
        self.label_expected = list(self.data['y'])

    def test_image_input_with_more_than_one_column(self):
        data = self.data.copy()
        data['img'] = [array.array('d', range(8))] * len(data)
        data['img'] = data['img'].pixel_array_to_image(2,2,2)
        self.assertRaises(ValueError, lambda: mxnet.io.SFrameIter(data, data_field=self.data_field + ['img']))
