import tensorflow as tf
import numpy as np
import os
import random
import cv2

def decorated_layer(layer):
	def wrapper(self, *args,  **kwargs):
		name = kwargs.setdefault('name', self.get_unique_name(layer.__name__))
		
		if len(self.inputs) == 1:
			ipt = self.inputs[0]
		else:
			ipt = self.inputs

		output = layer(self, ipt, *args, **kwargs)
		self.layers[name] = output

		print 'output shape', output.get_shape().as_list()
		return self.feed(output)

	return wrapper

class Network(object):
	def __init__(self, dataset = None, trainable = True): #data_shape = (height, width)

		self.__trainable = trainable
		if self.__trainable == True:
			assert not dataset == None
			self.__dataset = dataset

		self.__data_shape = dataset.shape
		self.__num_cls = dataset.num_cls
		self.__input_layer = tf.placeholder(tf.float32, shape = [None, self.__data_shape[0], self.__data_shape[1], self.__data_shape[2]], name = 'input')
		self.__image_info = tf.placeholder(tf.float32, shape = [None, self.__num_cls], name = "label")
		self.layers = {'input' : self.__input_layer, 'label' : self.__image_info}
		self.inputs = []
		
		self.setup()

	def feed(self,*args):
		assert len(args) != 0
		self.inputs = []

		for ipt in args:
			if isinstance(ipt, str):
				try:
					ipt = self.layers[ipt]
					print(ipt)
				except KeyError:
					print('Existing layers:',self.layers.keys())
					raise KeyError('Unknown layers %s' %ipt)
			else:
				print(ipt)
			self.inputs.append(ipt)
			
		return self

	def weight_variable(self, shape, stddev = 0.01):
		init = tf.truncated_normal(shape, stddev = stddev)
		return tf.Variable(init)

	def bias_variable(self, shape, value = 0.01):
		init = tf.constant(value, shape = shape)
		return tf.Variable(init)

	def __append(self, appendList, variables):
		if not appendList is None:
			assert isinstance(appendList, list)
			assert isinstance(variables, list)
			assert len(variables) > 0
			for item in variables:
				appendList.append(item)


	@decorated_layer
	def conv(self,input_data, k_w, k_d, s_w, s_h, name, relu = True, padding = 'SAME', value = 0.01, appendList = None):
		
		depth = input_data.get_shape().as_list()[-1]
		#print('input data depth of ' + name + ':', depth)
		#kernel
		kernel = self.weight_variable([k_w, k_w, depth, k_d], stddev = value)

		conv2d = tf.nn.conv2d(input_data, kernel, strides = [1, s_h, s_w, 1], padding = padding)

		bias = self.bias_variable([k_d], value = value)

		self.__append(appendList, [kernel, bias])

		if relu:
			return tf.nn.relu(tf.nn.bias_add(conv2d, bias), name = name)
		else:
			return tf.nn.bias_add(conv2d, bias, name = name)

	@decorated_layer
	def max_pooling(self, input_data, name, padding = 'SAME'):
		return tf.nn.max_pool(input_data, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],padding = padding, name = name)


	@decorated_layer
	def fc(self, input_data, output_dim, name, relu = True, value = 0.01, appendList = None):
		assert not isinstance(input_data, list)

		shape = input_data.get_shape().as_list()
		
		size = 1
		print 'shape', shape
		for i in shape[1:]:
			size *= i

		if len(shape) == 4:
			input_data = tf.reshape(input_data, [-1, size])
			#print(input_data.shape)
		
		w = self.weight_variable([size, output_dim], stddev = value)
		b = self.bias_variable([output_dim], value = value)

		self.__append(appendList, [w, b])

		if relu:
			op = tf.nn.relu_layer
		else:
			op = tf.nn.xw_plus_b

		return op(input_data, w, b, name = name)


	@decorated_layer
	def soft_max(self, input_data, name, loss = True):
		#print(input_data.get_shape().as_list())
		if loss:
			labels = self.layers['label']
			return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = input_data), name = name)
		else:
			return tf.nn.softmax(input_data, name = name)

	@decorated_layer
	def drop_out(self, input_data, name, keep_prob_name):
		kp = tf.placeholder(tf.float32, name = keep_prob_name)
		self.layers[keep_prob_name] = kp
		return tf.nn.dropout(input_data, keep_prob = kp, name = name)
		
	def setup(self):
		raise NotImplementedError('Function setup(self) must be implemented!')

			
	def train_model(self, epoch, ckpt_path):
		raise NotImplementedError('Function train_model(self, epoch, ckpt_path) must be implemented!')

			
	def im_detect(self, img):
		raise NotImplementedError('Function im_detect(self, img) must be implemented!')

	def get_unique_name(self, layer_name):
		count = len([name for name, _ in self.layers.items() if name.startswith(layer_name)])
		new_name = layer_name + '_' + str(count + 1)
		return new_name

