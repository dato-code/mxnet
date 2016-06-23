import mxnet as mx
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
    import graphlab
    __has_graphlab__ = True
except:
    pass


@unittest.skipIf(__has_sframe__ is False and __has_graphlab__ is False, 'graphlab or sframe not found')
class PretrainedImageClassifierTest(unittest.TestCase):
    def setUp(self):
        self.train = graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/mnist/sframe/train') 
        self.test = graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/mnist/sframe/test') 
        mx.pretrained_model.download_model('https://s3.amazonaws.com/dato-models/mxnet_models/release/image_classifier/mnist_lenet-1.0.tar.gz', overwrite=True)
        self.model = mx.pretrained_model.load_model('mnist_lenet') 

    def test_predict_topk(self):
	# Test accuracy of predictions
	preds = self.model.predict_topk(self.test, k=1)
	test = self.test.copy()
	test['preds'] = preds['label']
	accuracy = graphlab.evaluation.accuracy(test['label'], test['preds'])
	self.assertGreater(accuracy, 0.95)

        # Test SArray
	preds2 = self.model.predict_topk(self.test['image'], k=1)
        self.assertListEqual(list(preds['label']), list(preds2['label']))

        # Test Image
	preds3 = self.model.predict_topk(self.test['image'][0], k=1)
        self.assertEqual(preds['label'][0], preds3['label'][0])

	# Test k > 1		
	preds4 = self.model.predict_topk(self.test, k=5)
	self.assertEqual(len(preds) * 5, len(preds4))	

	# Test bad input
	duplicate_image = self.test.copy()
	duplicate_image['dup'] = duplicate_image['image']
	with self.assertRaises(TypeError):
            self.model.predict_topk(duplicate_image) 	

	# Test bad input
	no_image = self.test.copy()
	del no_image['image']
	with self.assertRaises(TypeError):
            self.model.predict_topk(no_image) 	


    def test_extract_features(self):
	# Test feature quality
	train_features = self.model.extract_feature(self.train)
	test_features = self.model.extract_feature(self.test)

	train = self.train.copy()
	train['features'] = train_features['feature']
	test = self.test.copy()
	test['features'] = test_features['feature']

	mdl = graphlab.logistic_classifier.create(train, features=['features'], target='label')
	accuracy = mdl.evaluate(test)['accuracy']

	self.assertGreater(accuracy, 0.95)
	
        # Test SArray
	test_features2 = self.model.extract_feature(self.test['image'])
	np.testing.assert_array_almost_equal(np.concatenate(list(test_features2['feature'])), np.concatenate(list(test_features['feature'])))
	
        # Test Image
	test_features3 = self.model.extract_feature(self.test['image'][0])
	np.testing.assert_array_almost_equal(np.concatenate(list(test_features3['feature'])), np.concatenate(list(test_features['feature'][0:1])))
	

	# Test bad input
	duplicate_image = self.test.copy()
	duplicate_image['dup'] = duplicate_image['image']
	with self.assertRaises(TypeError):
            self.model.extract_feature(duplicate_image) 	

	# Test bad input
	no_image = self.test.copy()
	del no_image['image']
	with self.assertRaises(TypeError):
            self.model.extract_feature(no_image) 	


    def test_extract_features_alias(self):
	features = np.concatenate(list(self.model.extract_feature(self.test)['feature']))
	alias_features = np.concatenate(list(self.model.extract_features(self.test)['feature']))

	np.testing.assert_array_almost_equal(features, alias_features)
	

