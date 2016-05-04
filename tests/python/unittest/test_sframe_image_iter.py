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
class SFrameImageIteratorBaseTest(unittest.TestCase):
    def setUp(self):
        (w, h, c, n) = (2, 4, 3, 10)
        self.images = [np.random.randint(256, size=(h,w,c)) for i in range(n)]
        self.data = gl.SFrame({'arr': [array.array('d', x.flatten()) for x in self.images],
                              'y': np.random.randint(2, size=n)})
        self.data['img'] = self.data['arr'].pixel_array_to_image(w, h, c)
        self.shape = (c, h, w)
        self.label_field = 'y'
        self.data_field = 'img'
        self.data_size = len(self.data)
        self.data_expected = list(x for arr in self.data['arr'] for x in arr)
        self.label_expected = list(self.data['y'])

    def test_one_batch(self):
        it = mxnet.io.SFrameImageIter(self.data, data_field=self.data_field,
                                 label_field=self.label_field,
                                 batch_size=self.data_size)
        label_actual = []
        data_actual = []
        for d in it:
            # reorder from (batch, channel, height, width) to (batch, height, width, channel)
            x = d.data[0].asnumpy()
            x = np.swapaxes(x, 1, 3)
            x = np.swapaxes(x, 1, 2)
            data_actual.extend(x.flatten())
            label_actual.extend(d.label[0].asnumpy().flatten())
        np.testing.assert_almost_equal(label_actual, self.label_expected)
        np.testing.assert_almost_equal(data_actual, self.data_expected)

    def test_type_check(self):
        self.assertRaises(lambda: mxnet.io.SFrameImageIter(self.data, data_field=[self.label_field],
                                              label_field=self.label_field,
                                              batch_size=self.data_size))

        self.assertRaises(lambda: mxnet.io.SFrameImageIter(self.data, data_field=[self.data_field, self.data_field],
                                              label_field=self.label_field,
                                              batch_size=self.data_size))

    def test_subtract_rgb(self):
        (mean_r, mean_g, mean_b, scale) = (1,2,3,0.5)
        it = mxnet.io.SFrameImageIter(self.data, data_field=self.data_field,
                                 label_field=self.label_field,
                                 batch_size=self.data_size,
                                 mean_r=mean_r, mean_g=mean_g, mean_b=mean_b, scale=scale)
        data_actual = []
        for d in it:
            # reorder from (batch, channel, height, width) to (batch, height, width, channel)
            x = d.data[0].asnumpy()
            x = np.swapaxes(x, 1, 3)
            x = np.swapaxes(x, 1, 2)
            x[:] /= scale
            x[:,:,:,0] += mean_r
            x[:,:,:,1] += mean_g
            x[:,:,:,2] += mean_b
            data_actual.extend(x.flatten())
        np.testing.assert_almost_equal(data_actual, self.data_expected)

        mean_nd = self.images[0]
        mean_nd[:,:,0] = mean_r
        mean_nd[:,:,1] = mean_g
        mean_nd[:,:,2] = mean_b
        it = mxnet.io.SFrameImageIter(self.data, data_field=self.data_field,
                                 label_field=self.label_field,
                                 batch_size=self.data_size,
                                 mean_nd=mean_nd, scale=scale)
        data_actual = []
        for d in it:
            # reorder from (batch, channel, height, width) to (batch, height, width, channel)
            x = d.data[0].asnumpy()
            x = np.swapaxes(x, 1, 3)
            x = np.swapaxes(x, 1, 2)
            x[:] /= scale
            x[:,:,:,0] += mean_r
            x[:,:,:,1] += mean_g
            x[:,:,:,2] += mean_b
            data_actual.extend(x.flatten())
        np.testing.assert_almost_equal(data_actual, self.data_expected)
