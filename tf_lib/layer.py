import tensorflow as tf

def WeightVariable(shape, reg_list = None, var_name = None):
	weight = tf.Variable(tf.truncated_normal(shape, stddev = 0.1), \
											name = var_name)

	if(reg_list is not None):
		reg_list.append(weight)

	return weight

def BiasVariable(value, shape, var_name = None):
	return tf.Variable(tf.constant(value, shape = shape), name = 'bias')

def NormVariable(shape, ofs_name = None, sc_name = None):
	offset = tf.Variable(tf.constant(0.1, shape = shape), name = ofs_name)
	scale = tf.Variable(tf.ones(shape), name = sc_name)
	return offset, scale

def Conv(input_tensor, filter_shape, strides = [1, 1, 1, 1], padding = 'VALID', \
		bias = False, data_format = 'NHWC', reg_list = None, name = None):
	# name
	if(name):
		filter_name = name + '/filter'
		bias_name = name + '/bias'
	else:
		filter_name = None
		bias_name = None

	# conv
	conv_filter = \
		WeightVariable(filter_shape, reg_list = reg_list, var_name = filter_name)

	conv_res = tf.nn.conv2d(input_tensor, conv_filter, strides = strides, \
				padding = padding, data_format = data_format, name = name)

	# bias
	if(bias is False):
		return conv_res
	else:
		if(data_format == 'NHWC'):
			channel_num = filter_shape[3]
		else: #'NCWH'
			channel_num = filter_shape[1]

		conv_bias = BiasVariable(0.1, [channel_num], bias_name)

		return tf.nn.bias_add(conv_res, conv_bias)

def FC(input_tensor, weights, biases):
	return tf.matmul(input_tensor, weights) + biases

def GLU(input_tensor):
	tensor_a, tensor_b = tf.split(input_tensor, 2, 3)
	return tensor_a * tf.nn.sigmoid(tensor_b)

def BatchNorm(input_tensor, shape, is_test, iteration, name = None):
	# name
	if(name):
		offset_name = name + '/offset'
		scale_name = name + '/scale'
	else:
		offset_name = None
		scale_name = None

	offset, scale = NormVariable(shape, offset_name, scale_name)
	ema = tf.train.ExponentialMovingAverage(0.998, iteration)

	mean, variance = tf.nn.moments(input_tensor, [0])
	update_moving_average = ema.apply([mean, variance])

	m = tf.cond(is_test, lambda: ema.average(mean), lambda: mean)
	v = tf.cond(is_test, lambda: ema.average(variance), lambda: variance)

	output_tensor = \
		tf.nn.batch_normalization(input_tensor, m, v, offset, scale, 0.001)

	return output_tensor, update_moving_average

def LayerNormMoments(x, axes = 1, scope = None, epsilon = 0.001):
	with tf.name_scope(scope):
		mean = tf.reduce_mean(x, axes, keep_dims = True)

		variance = tf.sqrt(tf.reduce_mean(\
			tf.square(x - mean), axes, keep_dims = True) + epsilon)

	return mean, variance

# input_tensor: [m, d]
# offset: [d]
# scale: [d]
# output_tensor: [m, d]
def LayerNorm(input_tensor, scope):
	with tf.name_scope(scope):
		input_tensor_shape = input_tensor.get_shape().as_list()
		ln_shape = [input_tensor_shape[1]]

		offset, scale = NormVariable(ln_shape, 'offset', 'scale')
		mean, variance = LayerNormMoments(input_tensor)

		output_tensor = (scale * (input_tensor - mean)) / variance + offset

	return output_tensor
