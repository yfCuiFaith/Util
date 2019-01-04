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

def PlainBlock(input_tensor, filter_shape, is_test, g_step, \
		update_ema_gp, block_name):
	with tf.name_scope(block_name):
		input_tensor_shape = input_tensor.get_shape().as_list()
		batch_size = input_tensor_shape[0]

		padding = filter_shape[0] / 2

		conv_name = '%s_conv' % block_name
		ln_name = '%s_ln' % block_name

		block_input = tf.reshape(input_tensor, [-1, input_tensor_shape[3]])
		block_ln = LayerNorm(block_input, scope = ln_name)
		block_ln = tf.reshape(block_ln, [batch_size, -1, 1, input_tensor_shape[3]])
		block_glu = GLU(block_ln)

		block_glu = tf.pad(block_glu, [[0, 0], [padding, padding], [0, 0], [0, 0]])

		block_conv = Conv(block_glu, filter_shape, name = conv_name)

	return block_conv

def ResidualBlock(input_tensor, filter_shape, is_test, g_step, \
		update_ema_gp, block_name):
	with tf.name_scope(block_name):
		input_tensor_shape = input_tensor.get_shape().as_list()
		batch_size = input_tensor_shape[0]

		padding = filter_shape[0] / 2

		conv_name = '%s_conv' % block_name
		ln_name = '%s_ln' % block_name

		block_input = tf.reshape(input_tensor, [-1, input_tensor_shape[3]])
		block_ln = LayerNorm(block_input, scope = ln_name)
		block_ln = tf.reshape(block_ln, [batch_size, -1, 1, input_tensor_shape[3]])
		block_glu = GLU(block_ln)
		
		block_glu = tf.pad(block_glu, [[0, 0], [padding, padding], [0, 0], [0, 0]])

		block_conv = Conv(block_glu, filter_shape, name = conv_name)

		block_output = input_tensor + block_conv
	return block_output


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

def NormBlock(input_tensor, is_test, g_step, update_ema_gp, block_name):
	with tf.name_scope(block_name):
		input_tensor_shape = input_tensor.get_shape().as_list()
		batch_size = input_tensor_shape[0]

		conv_name = '%s_conv' % block_name
		ln_name = '%s_ln' % block_name

		block_input = tf.reshape(input_tensor, [-1, input_tensor_shape[3]])
		block_ln = LayerNorm(block_input, scope = ln_name)
		block_ln = tf.reshape(block_ln, [batch_size, -1, 1, input_tensor_shape[3]])
		block_glu = GLU(block_ln)
	return block_glu

def StageBlock(input_tensor, filter_shape, layer_num, is_last, \
		is_test, g_step, update_ema_gp, block_name):
	with tf.name_scope(block_name):
		buffer_tensor = input_tensor

		for i in range(layer_num):
			sub_block_name = str.format('%s_block%d' % (block_name, i))

			buffer_tensor = \
				ResidualBlock(buffer_tensor, filter_shape, \
					is_test, g_step, update_ema_gp, sub_block_name)
			print(buffer_tensor)
		if(is_last):
			buffer_tensor = \
				NormBlock(buffer_tensor, is_test, g_step, \
					update_ema_gp, '%s_top' % (block_name))
		else:
			buffer_tensor = \
				PlainBlock(buffer_tensor, filter_shape, is_test, \
					g_step, update_ema_gp, '%s_top' % (block_name))
		print(buffer_tensor)
	return buffer_tensor

def DevPlainBlock(input_tensor, filter_shape, direction, block_name):
	with tf.name_scope(block_name):
		input_tensor_shape = input_tensor.get_shape().as_list()
		batch_size = input_tensor_shape[0]

		conv_name = '%s_conv' % block_name
		ln_name = '%s_ln' % block_name

		block_input = tf.reshape(input_tensor, [-1, input_tensor_shape[3]])
		block_ln = LayerNorm(block_input, scope = ln_name)
		block_ln = tf.reshape(block_ln, [batch_size, -1, 1, input_tensor_shape[3]])
		block_glu = GLU(block_ln)

		block_conv = DevConv(block_glu, filter_shape, direction, name = conv_name)

	return block_conv

def DevResidualBlock(input_tensor, filter_shape, direction, block_name):
	with tf.name_scope(block_name):
		input_tensor_shape = input_tensor.get_shape().as_list()
		batch_size = input_tensor_shape[0]

		conv_name = '%s_conv' % block_name
		ln_name = '%s_ln' % block_name

		block_input = tf.reshape(input_tensor, [-1, input_tensor_shape[3]])
		block_ln = LayerNorm(block_input, scope = ln_name)
		block_ln = tf.reshape(block_ln, [batch_size, -1, 1, input_tensor_shape[3]])
		block_glu = GLU(block_ln)
		
		block_conv = DevConv(block_glu, filter_shape, direction, name = conv_name)

		block_output = input_tensor + block_conv
	return block_output
