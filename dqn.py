import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import tensorflow as tf
import numpy as np
from network import Network
from collections import deque
import cv2
import random
import os

IMAGE_SIZE = (80,80)
BATCH_SIZE = 30
GAMMA = 0.99
EBSILON = 0.02
UPDATE_STEP = 100
FRAME_PER_ACTION = 1
HOLD_ACTION = False
REPLAY_MEMORY_SIZE = 50000
EXPLORE_STEPS = 200000
LEARNING_RATE = 1e-6
CKPT_PATH = os.path.join(os.getcwd(), 'model', 'network-dqn')
SAVE_STEP = 10000

class DQN(Network):
	def __init__(self, num_action, dataset = None, trainable = True): #data_shape = (height, width)
		assert dataset == None
		self.__data_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 4)
		self.__trainable = trainable
		self.__replay_memory = deque()
		self.__num_action = num_action

		self.__train_input = tf.placeholder(tf.float32, shape = [None, self.__data_shape[0], self.__data_shape[1], self.__data_shape[2]], name = 'train_input')
		self.__target_input = tf.placeholder(tf.float32, shape = [None, self.__data_shape[0], self.__data_shape[1], self.__data_shape[2]], name = 'target_input')

		self.layers = {'train_input' : self.__train_input, 'target_input' : self.__target_input}

		self.inputs = []
		self.__session = tf.InteractiveSession()
		self.__optimizer = None
		

		#list used to store two network layes
		self.__train_variables = []
		self.__target_variables = []

		self.__current_state = None

		self.__last_action = [0 for _ in range(num_action)]
		self.__time_step = 0
		self.__ebsilon = EBSILON
		
		self.setup()

	def setup(self):
		#train network
		print 'Training network'
		(self.feed('train_input')
		.conv(8, 32, 4, 4, name = 'conv_train_1', appendList = self.__train_variables).max_pooling(name = 'max_pool_train_1')
		.conv(4, 64, 2, 2, name = 'conv_train_2', appendList = self.__train_variables)
		.conv(3, 64, 1, 1, name = 'conv_train_3', appendList = self.__train_variables)
		.fc(512, name = 'fc_train_1', appendList = self.__train_variables)
		.fc(self.__num_action, name = 'QValues', relu = False, appendList = self.__train_variables))


		#target network
		print 'Target network'
		(self.feed('target_input')
		.conv(8, 32, 4, 4, name = 'conv_target_1', appendList = self.__target_variables).max_pooling(name = 'max_pool_target_1')
		.conv(4, 64, 2, 2, name = 'conv_target_2', appendList = self.__target_variables)
		.conv(3, 64, 1, 1, name = 'conv_target_3', appendList = self.__target_variables)
		.fc(512, name = 'fc_target_1', appendList = self.__target_variables)
		.fc(self.__num_action, name = 'QValues_target', relu = False, appendList = self.__target_variables))

		

		#loss function
		print 'Loss function'
		self.layers['action_input'] = tf.placeholder(tf.float32, shape = [None, self.__num_action], name = 'action_input')
		qvalues = tf.reduce_sum(tf.multiply(self.layers['QValues'], self.layers['action_input'] ),  reduction_indices = 1)
		self.layers['y'] = tf.placeholder(tf.float32, shape = [None], name = 'target_values')
		self.layers['loss_function'] = tf.reduce_mean(tf.square(self.layers['y'] - qvalues))
		self.__optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.layers['loss_function'])

		self.__session.run(tf.global_variables_initializer())
		self.__saver = tf.train.Saver()

		ckpt = tf.train.get_checkpoint_state('model')

		if ckpt and ckpt.model_checkpoint_path:
				self.__saver.restore(self.__session, ckpt.model_checkpoint_path)
				print 'Load trained model %s' %ckpt.model_checkpoint_path
		else:
				print 'No pretrained model found' 

		


	def copy_network(self):
		for i in range(len(self.__train_variables)):
			self.__session.run(self.__target_variables[i].assign(self.__train_variables[i]))

	def preprocessing(self, img):
		img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), IMAGE_SIZE)
		ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
		img = np.reshape(img, [IMAGE_SIZE[0], IMAGE_SIZE[1], 1])
		return img

	def get_action(self):
		action = np.zeros(self.__num_action)

		if self.__time_step % FRAME_PER_ACTION == 0:
			if random.random() < self.__ebsilon:
				index = random.randint(0, self.__num_action - 1)
			else:
				qvalue = self.layers['QValues'].eval(feed_dict = {self.layers['train_input']:[self.__current_state]})[0]
				index = np.argmax(qvalue)

			action[index] = 1

		elif HOLD_ACTION:
			action =  self.__last_action
		else:
			action[0] = 1
		return action

	def set_initial_state(self, observation):
		observation = np.reshape(self.preprocessing(observation), IMAGE_SIZE)
		self.__current_state = np.stack((observation, observation, observation, observation), axis = 2)

	def train_model(self, epoch, ckpt_path):
		pass

	def iterate(self, observation, action, reward, terminate):
		observation = self.preprocessing(observation)
		newState= np.append(self.__current_state[:, :, 1:], observation, axis = 2)

		#make experience
		experience = (self.__current_state, action, newState, reward, terminate)
		self.__replay_memory.append(experience)

		self.__current_state = newState

		if self.__time_step > BATCH_SIZE:
			self.train_model()

		#reduce the explore rate
		self.__time_step += 1
		if self.__ebsilon >= 0 and self.__time_step <= EXPLORE_STEPS:
			self.__ebsilon -= EBSILON / EXPLORE_STEPS

		#remove the old experience if the replay memory is too large
		if len(self.__replay_memory) > REPLAY_MEMORY_SIZE:
			self.__replay_memory.popleft()

		#save the model
		if self.__time_step % SAVE_STEP == 0:
			self.__saver.save(self.__session, CKPT_PATH, global_step = self.__time_step)

		if self.__time_step % UPDATE_STEP == 0:
			print 'Time step: %s, Explore rate: %s' %(self.__time_step, self.__ebsilon) 
			self.copy_network()

		
	def train_model(self):
		mini_batch = random.sample(self.__replay_memory, BATCH_SIZE)
		state = [item[0] for item in mini_batch]
		action = [item[1] for item in mini_batch]
		newState = [item[2] for item in mini_batch]
		reward = [item[3] for item in mini_batch]
		terminate = [item[4] for item in mini_batch]

		new_q = self.layers['QValues_target'].eval(feed_dict = {self.layers['target_input']:newState})
		y = []
		for i, q_value in enumerate(new_q):
			if terminate[i] == True:
				y.append(reward[i])
			else:
				y.append(reward[i] + GAMMA * np.max(q_value))

		self.__optimizer.run(feed_dict = {self.layers['train_input']:state, self.layers['y']:y, self.layers['action_input']:action})

if __name__ == '__main__':
	num_action = 2
	dqn = DQN(num_action)
	flappyBird = game.GameState()

	action0 = np.array([1,0])  # do nothing
	observation0, reward0, terminal = flappyBird.frame_step(action0)
	dqn.set_initial_state(observation0)

	while True:
		action = dqn.get_action()
		nextObservation,reward,terminal = flappyBird.frame_step(action)
		dqn.iterate(nextObservation, action, reward, terminal)

