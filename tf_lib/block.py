import tensorflow as tf
import sys
import os
sys.path.append(os.getcwd())
from layer import *

def BaseBlockUnit(input_tensor, filter_shape, reg_list, conv_name, \
		bn_shape, is_test, g_step, bn_name, no_relu = False):
	conv_input = tf.pad(input_tensor, [[0, 0], [1, 1], [0, 0], [0, 0]])
	conv = Conv(conv_input, filter_shape, reg_list = reg_list, name = conv_name)
	bn, update_ema = \
		BatchNorm(conv, bn_shape, is_test, g_step, bn_name)
	
	if(no_relu == False):
		relu = tf.nn.relu(bn)
		return relu, update_ema
	return bn, update_ema

def BaseBlock(input_tensor, filter_shape, bn_shape, is_test, g_step, \
		reg_list, update_ema_gp, block_name):
	with tf.name_scope(block_name):
		block_relu1, block_update_ema1 = \
			BaseBlockUnit(input_tensor, filter_shape, reg_list, 'conv1', \
				bn_shape, is_test, g_step, 'bn1')

		block_relu2, block_update_ema2 = \
			BaseBlockUnit(block_relu1, filter_shape, reg_list, 'conv2', \
				bn_shape, is_test, g_step, 'bn2')

		block_concat1 = input_tensor + block_relu2

		block_relu3, block_update_ema3 = \
			BaseBlockUnit(block_relu2, filter_shape, reg_list, 'conv3', \
				bn_shape, is_test, g_step, 'bn3')

		block_relu4, block_update_ema4 = \
			BaseBlockUnit(block_relu3, filter_shape, reg_list, 'conv4', \
				bn_shape, is_test, g_step, 'bn4')

		block_concat2 = block_concat1 + block_relu4

	update_ema_gp.extend([block_update_ema1, block_update_ema2, \
		block_update_ema3, block_update_ema4])
	
	return block_concat2

'''
def ResidualBlock(input_tensor, filter_shape, is_test, g_step, \
		reg_list, update_ema_gp, block_name):
	with tf.name_scope(block_name):
		bn_shape = [1, 1, filter_shape[3]]

		conv1_input = tf.reshape(input_tensor, [1, -1, filter_shape[1], filter_shape[2]])
		conv1_input = tf.pad(conv1_input, [[0, 0], [1, 1], [0, 0], [0, 0]])

		block_conv1 = Conv(conv1_input, filter_shape, reg_list = reg_list, name = 'conv1')
		block_conv1 = tf.reshape(block_conv1, [-1, 1, 1, filter_shape[3]])
		block_bn1, block_update_ema1 = \
			BatchNorm(block_conv1, bn_shape, is_test, g_step, 'bn1')
		block_bn1 = tf.reshape(block_bn1, [1, -1, 1, filter_shape[3]])
		block_relu1 = tf.nn.relu(block_bn1)

		conv2_input = tf.reshape(block_relu1, [1, -1, filter_shape[1], filter_shape[2]])
		conv2_input = tf.pad(conv2_input, [[0, 0], [1, 1], [0, 0], [0, 0]])

		block_conv2 = Conv(conv2_input, filter_shape, reg_list = reg_list, name = 'conv2')
		block_conv2 = tf.reshape(block_conv2, [-1, 1, 1, filter_shape[3]])
		block_bn2, block_update_ema2 = \
			BatchNorm(block_conv2, bn_shape, is_test, g_step, 'bn2')
		block_bn2 = tf.reshape(block_bn2, [1, -1, 1, filter_shape[3]])
		block_concat = block_bn2 + input_tensor
		block_relu2 = tf.nn.relu(block_concat)
	update_ema_gp.extend([block_update_ema1, block_update_ema2])
	return block_relu2
'''

def ResidualBlock(input_tensor, filter_shape, is_test, g_step, \
		reg_list, update_ema_gp, block_name):
	with tf.name_scope(block_name):
		bn_shape = [1, 1, filter_shape[3]]

		#conv1_input = tf.reshape(input_tensor, [1, -1, filter_shape[1], filter_shape[2]])
		conv1_input = tf.pad(input_tensor, [[0, 0], [1, 1], [0, 0], [0, 0]])

		block_conv1 = Conv(conv1_input, filter_shape, reg_list = reg_list, name = 'conv1')
		block_conv1 = tf.reshape(block_conv1, [-1, filter_shape[3]])
		block_ln1 = LayerNorm(block_conv1, 'ln1')
		block_ln1 = tf.reshape(block_ln1, [1, -1, 1, filter_shape[3]])
		#block_relu1 = tf.nn.relu(block_ln1)
		block_glu1 = GLU(block_ln1)
		block_concat1 = block_glu1 + input_tensor

		#conv2_input = tf.reshape(block_relu1, [1, -1, filter_shape[1], filter_shape[2]])
		conv2_input = tf.pad(block_concat1, [[0, 0], [1, 1], [0, 0], [0, 0]])

		block_conv2 = Conv(conv2_input, filter_shape, reg_list = reg_list, name = 'conv2')
		block_conv2 = tf.reshape(block_conv2, [-1, filter_shape[3]])
		block_ln2 = LayerNorm(block_conv2, 'ln2')
		block_ln2 = tf.reshape(block_ln2, [1, -1, 1, filter_shape[3]])
		#block_relu2 = tf.nn.relu(block_concat)
		block_glu2 = GLU(block_ln2)
		block_concat2 = block_glu2 + block_concat1
	
	return block_glu2
