import os as _os
import logging as _logging
import json as _json
from collections import namedtuple as _namedtuple
from .model import FeedForward as _FeedForward
from .model import extract_feature as _extract_feature
from . import ndarray as _ndarray
from . import io as _io
import numpy as _np
import requests

_LOGGER = _logging.getLogger(__name__)

class ModelTypeEnum(object):
    """
    Enumeration of Pretrained Model Type
    """
    IMAGE_CLASSIFIER= 'IMAGE_CLASSIFIER'

ModelEntry = _namedtuple('ModelEntry', ['name', 'type', 'version'])


"""
Default location for model download
"""
DEFAULT_MODEL_LOCATION=_os.path.expanduser('~/.graphlab/mxnet_models')


def list_models(location=DEFAULT_MODEL_LOCATION):
    """
    Return list of pretrained model names.

    Parameters
    ----------
    location : str, optional
      The local directory where the model is saved to.

    Examples
    --------
    >>> mx.pretrained_model.list_models()
    """
    if not _os.path.exists(location):
        _os.makedirs(location)

    models = [p for p in _os.listdir(location)]
    ret = []
    for name in models:
        model_path = _os.path.join(location, name)
        if not _os.path.isdir(model_path):
            continue
        metadata_path = _os.path.join(model_path, 'metadata.json')
        version = None
        try:
            f = open(metadata_path)
            metadata = _json.load(f)
            version = metadata['version']
            model_type = metadata['model_type']
            name = metadata['name']
            ret.append(ModelEntry(name, model_type, version))
        except Exception as e:
            _LOGGER.warn('Unable to open or parse model metadata %s.' % metadata_path)
    return ret

def _download_file(url, target_file):
    r = requests.get(url, stream=True)
    with open(target_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def download_model(url, location=DEFAULT_MODEL_LOCATION, overwrite=False):
    """
    Perform downloading the model to local filesystem.

    Parameters
    ----------
    url : str,
      URL of the model. Get the list of available models from https://github.com/dato-code/mxnet.
    location : str, optional
      The local directory where the model is saved to.
    overwrite : bool, optional
      If true, remove existing models first.

    Examples
    --------
    >>> mx.pretrained_model.download_model('https://.../InceptionV3.tar.gz')
    >>> model = mx.pretrained_model.load_model('InceptionV3')
    """
    name = url.split('/')[-1]
    name = name.split('-')[0]
    target_dir = _os.path.join(location, name)
    target_exists = _os.path.exists(target_dir)
    if overwrite is False and target_exists:
        raise OSError("Target directory %s already exists. Please remove existing model or set overwrite to True" % target_dir)
    # Peform download
    _LOGGER.info("Begin downloading %s" % (url))
    import tempfile
    import hashlib
    import shutil
    import tarfile
    import sys

    f = tempfile.NamedTemporaryFile(suffix='tar.gz')
    _download_file(url, f.name)
    _LOGGER.info("Filed downloaded to %s" % (f.name))

    # Extract
    if target_exists:
        _LOGGER.info("Remove existing model: %s" % target_dir)
        shutil.rmtree(target_dir)
    _os.makedirs(target_dir)
    _LOGGER.info("Extracting model to %s" % target_dir)
    tar = tarfile.open(f.name)
    tar.extractall(location)
    tar.close()
    _LOGGER.info("Model %s is downloaded to %s" % (name, target_dir))


def load_model(name, ctx=None, location=DEFAULT_MODEL_LOCATION):
    """
    Load a pretrained model by name.

    Parameters
    ----------
    name : str
        Name of the pretrained model. Model must be downloaded first
    ctx : mx.context, optional
        Context of the model. Default None is equivalent to mx.cpu()
    location : str, optional
        The directory which contains downloaded models

    Examples
    --------
    >>> model = mx.pretrained_model.load_model('InceptionV3', ctx=mx.gpu(0))
    """
    if not any(name == m.name for m in list_models(location)):
        raise KeyError('Model %s does not exist. Models can be listed using list_models()' % name)
    target_dir = _os.path.join(location, name)
    return load_path(target_dir, ctx)


def load_path(target_dir, ctx=None):
    """
    Load a pretrained model by path.

    Parameters
    ----------
    path : str
        Path of the downloaded pretrained model.
    ctx : mx.context, optional
        Context of the model. Default None is equivalent to mx.cpu().
    location : str, optional
        The directory which contains downloaded pretrained models

    Examples
    --------
    >>> model = mx.pretrained_model.load_model('~/.graphlab/mxnet_models/InceptionV3')
    """

    _LOGGER.debug('load from: %s' % (target_dir))
    target_dir = _os.path.expanduser(target_dir)

    # Load the model metadata
    metadata_path = _os.path.join(target_dir, 'metadata.json')
    _LOGGER.debug('metadata_path: %s' % metadata_path)
    metadata = _json.load(open(metadata_path))
    _LOGGER.debug('metadata: %s' % str(metadata))
    if metadata['mean_nd'] is not None:
        mean_nd_path = _os.path.join(target_dir, metadata['mean_nd'])
        mean_nd = _ndarray.load(mean_nd_path)['mean_img'].asnumpy()
        # mean_nd is in c, h, w order. need to convert to h, w, c
        mean_nd = _np.swapaxes(mean_nd, 0, 2) # c, h, w -> w, h, c
        mean_nd = _np.swapaxes(mean_nd, 0, 1) # w, h, c -> h, w, c
        metadata['mean_nd'] = mean_nd

    # Load Image Classifier
    if metadata['model_type'] == ModelTypeEnum.IMAGE_CLASSIFIER:
        param_file = [f for f in _os.listdir(target_dir) if f.endswith('.params')]
        if len(param_file) != 1:
            raise ValueError('Invalid model directory %s. Please remove the directory and redownload the model' % target_dir)

        # Parse the file name to get prefix and epoch
        _LOGGER.debug('param_file: %s' % param_file[0])
        prefix = _os.path.splitext(param_file[0])[0]
        epoch = prefix.split('-')[-1]
        prefix = prefix[:-(len(epoch) + 1)]
        prefix = _os.path.join(target_dir, prefix)
        epoch = int(epoch)
        _LOGGER.debug('prefix: %s, epoch: %s' % (prefix, epoch))

        # Load the feedforward model
        model = _FeedForward.load(prefix, epoch, ctx)

        # Load the labels
        label_file = _os.path.join(target_dir, 'labels.json')
        _LOGGER.debug('label_file: %s' % label_file)
        labels = _json.load(open(label_file))
        return ImageClassifier(model, labels, metadata)
    else:
        raise TypeError('Unexpected model type: %s', metadata['model_type'])


class ImageClassifier(object):
    """
    Wrapper of pretrained image classifier model.

    Use :py:func:`load_model` or :py:func:`load_path` to load the model. Do not
    construct directly.

    Parameters
    ----------
    model :  FeedForward
        The underlying model
    labels : list
        Map from index to label
    metadata : dict
        Metadata of the model including name, version, input shape, etc.
    """
    def __init__(self, model, labels, metadata):
        self._model = model
        self._labels = labels
        self._input_shape = metadata['input_shape']
        self.mean_nd = metadata['mean_nd']
        self.mean_rgb = metadata['mean_rgb']
        self.rescale = metadata['scale']
        self._name = metadata['name']
        self._version = metadata['version']

    @property
    def model(self):
        return self._model

    @property
    def labels(self):
        return self._labels

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def __str__(self):
        return "ImageClassifier: " + self._name + "(version %s)" % self._version

    def __repr__(self):
        return self.__str__()

    def _make_dataiter(self, data, batch_size):
        try:
            import graphlab as _gl
        except ImportError:
            import sframe as _gl
        except ImportError:
            raise ImportError('Require GraphLab Create or SFrame')

        if type(data) is not _gl.SFrame and type(data) is not _gl.SArray:
            raise TypeError('Input data must be SFrame or SArray')
        if type(data) is _gl.SArray and data.dtype() != _gl.Image:
            raise TypeError('Expect image typed SArray, actual type is %s' % str(data.dtype()))
        elif type(data) is _gl.SFrame:
            if len(data.column_names()) != 1 or data.column_types()[0] != _gl.Image:
                raise TypeError('Input SFrame must contain only a single image typed column')
        if batch_size < len(data):
            batch_size = len(data)

        if type(data) is _gl.SArray:
            data = _gl.SFrame({'image': data})
        image_col = data.column_names()[0]
        first_image = data[image_col][0]
        input_shape = (first_image.height, first_image.width, first_image.channels)

        if input_shape != tuple(self.input_shape):
            _LOGGER.info('Detect image shape mismatches network input shape. Perform resize to shape %s' \
                    % str(tuple(self.input_shape)))
            data_resize = _gl.SFrame()
            data_resize[image_col] = _gl.image_analysis.resize(data[image_col],
                                                               self.input_shape[0],
                                                               self.input_shape[1],
                                                               self.input_shape[2], decode=True)
            data = data_resize

        dataiter = _io.SFrameImageIter(data, data_field=[image_col],
                                       batch_size=batch_size,
                                       mean_r=self.mean_rgb[0],
                                       mean_g=self.mean_rgb[1],
                                       mean_b=self.mean_rgb[2],
                                       mean_nd=self.mean_nd,
                                       scale=self.rescale)
        return dataiter


    def extract_feature(self, data, batch_size=50):
        # Make DataIter
        dataiter = self._make_dataiter(data, batch_size)
        return _extract_feature(self.model, dataiter)


    def predict_topk(self, data, k=5, batch_size=50):
        """
        Predict the topk classes for given data

        Parameters
        ----------
        data : SFrame or SArray[Image]
            SFrame of a single image typed column.
            Images must have the same size as the model's input shape.
        k : int, optional
            Number of classes returned for each input
        batch_size : int, optional
            batch size of the input to the internal model. Larger
            batch size makes the prediction faster but requires more memory.

        Returns
        -------
        out : SFrame
            An SFrame of 5 columns: row_id, label_id, label, score, rank

        Examples
        --------
        >>> m = mx.pretrained_model.load_model('MNIST_Conv')
        >>> sf = SFrame('http://s3.amazonaws.com/dato-datasets/mnist/sframe/train')
        >>> m.predict_topk(sf['image'])
        """
        try:
            import graphlab as _gl
        except ImportError:
            import sframe as _gl
        except ImportError:
            raise ImportError('Require GraphLab Create or SFrame')

        # Check input
        if k > len(self.labels):
            k = len(self.labels)

        # Make DataIter
        dataiter = self._make_dataiter(data, batch_size)

        # Make prediction
        # pred[i][j] is the score of row i belongs to label j
        pred = self.model.predict(dataiter)
        # top_idx[i][k] is the label index of kth highest score of row i
        top_idx = pred.argsort()[:,-k:][:,::-1]
        # Take row wise index, to get topk score for each row
        top_scores = pred[_np.arange(pred.shape[0])[:,None], top_idx]

        top_labels = [self.labels[i] for i in top_idx.flatten()]
        row_ids = _np.repeat(_np.arange(len(pred)), k)
        ranks = _np.tile(_np.arange(1, k+1), len(pred))

        ret = _gl.SFrame()
        ret['row_id'] = row_ids
        ret['label_id'] = top_idx.flatten()
        ret['label'] = top_labels
        ret['score'] = top_scores.flatten()
        ret['rank'] = ranks
        return ret
