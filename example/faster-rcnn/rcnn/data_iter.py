import mxnet as mx
import numpy as np
import minibatch


class VOCIter(mx.io.DataIter):
    def __init__(self, voc, max_gt_box, shuffle=False, rand_flip=False, mode='train'):
        """
        This Iter will provide roi data to Faster R-CNN network
        :param roidb: must be preprocessed
        :param batch_size: must be 1
        :param shuffle: bool
        :return: VOCIter
        """
        super(VOCIter, self).__init__()

        self.voc = voc
        self.batch_size = 1
        self.shuffle = shuffle
        self.mode = mode
        self.max_gt_box = max_gt_box

        self.cur = 0
        self.size = voc.size()
        self.index = np.arange(self.size)

        # data
        self.batch = None
        self.data = None
        self.gt_boxes = None
        self.gt_pad = None

        self.get_batch()
        self.data_name = self.data.keys()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self.data.items()]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=None,
                                   pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def getindex(self):
        return self.cur

    def getpad(self):
        return 0

    def get_batch(self):
        if self.mode == 'train':
            self.batch = self._get_train_batch()
            self.data = {'data': self.batch['data'],
                         'im_info': self.batch['im_info'],
                         'gt_boxes': self.batch['gt_boxes'],
                         'gt_pad': self.batch['gt_pad']}
        else:
            self.batch = self._get_test_batch()
            self.data = {'data': self.batch['data'],
                         'im_info': self.batch['im_info']}

    def _get_train_batch(self):
        """
        utilize minibatch sampling, e.g. 2 images and 64 rois per image
        :return: training batch (e.g. 128 samples)
        """
        idx = self.voc.at(self.cur)
        batch = minibatch.get_minibatch(self.voc, idx, self.max_gt_box)
        return batch

    def _get_test_batch(self):
        """
        testing batch is composed of 1 image, all rois
        :return: testing batch
        """
        idx = self.voc.at(self.cur)
        batch = minibatch.get_testbatch(self.voc, idx)
        return batch
