import tensorflow as tf

def WeightVariable(shape, scope = None, var_name = None, regularizer = 'l2_weights'):
	weight = tf.Variable(tf.truncated_normal(shape, stddev = 0.1), \
			name = var_name)

	if(regularizer is not None):
		tf.add_to_collection(regularizer, weight)

	return weight

def BiasVariable(value, shape, scope = None, var_name = None):
	return tf.Variable(tf.constant(value, shape = shape), name = 'bias')

def NormVariable(shape, scope = None, ofs_name = None, sc_name = None):
	offset = tf.Variable(tf.zeros(shape), name = ofs_name)
	scale = tf.Variable(tf.ones(shape), name = sc_name)
	return offset, scale

def Conv(input_tensor, filter_shape, strides = [1, 1, 1, 1], padding = 'VALID', \
		bias = False, data_format = 'NHWC', name = None):
	# name
	scope = name + '_w'
	filter_name = 'filter'
	bias_name = 'bias'

	# conv
	conv_filter = WeightVariable(filter_shape, scope, filter_name)

	conv_res = tf.nn.conv2d(input_tensor, conv_filter, strides = strides, \
				padding = padding, data_format = data_format, name = name)

	# bias
	if(bias is False):
		return conv_res
	else:
		if(data_format == 'NHWC'):
			channel_num = filter_shape[3]
		else: #'NCHW'
			channel_num = filter_shape[1]

		conv_bias = BiasVariable(0.1, [channel_num], scope, bias_name)

		return tf.nn.bias_add(conv_res, conv_bias)

def FC(input_tensor, weights, biases):
	return tf.matmul(input_tensor, weights) + biases

def GLU(input_tensor):
	tensor_a, tensor_b = tf.split(input_tensor, 2, 3)
	return tensor_a * tf.nn.sigmoid(tensor_b)

def BatchNorm(input_tensor, shape, is_test, iteration, name = None):
	# name
	scope = name + '_w'
	offset_name = 'offset'
	scale_name = 'scale'

	offset, scale = NormVariable(shape, scope, offset_name, scale_name)
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

		offset, scale = NormVariable(ln_shape, scope, 'offset', 'scale')
		mean, variance = LayerNormMoments(input_tensor)

		output_tensor = (scale * (input_tensor - mean)) / variance + offset

	return output_tensor

def Align(h, z, filter_shape, name = None):
	padding = filter_shape[0] / 2

	h_ = tf.pad(h, [[0, 0], [padding, padding], [0, 0], [0, 0]])
	d = Conv(h_, filter_shape, bias = True, name = name)

	d_tensor_shape = d.get_shape().as_list()
	z_tensor_shape = z.get_shape().as_list()

	d = tf.reshape(d, [-1, d_tensor_shape[3]])
	z = tf.reshape(z, [-1, z_tensor_shape[3]])

	#d = LayerNorm(d, name + '_ln0')
	#z = LayerNorm(z, name + '_ln1')

	d = tf.nn.l2_normalize(d, 1)
	z = tf.nn.l2_normalize(z, 1)

	d = tf.reshape(d, [d_tensor_shape[0], -1, d_tensor_shape[3]])
	z = tf.reshape(z, [z_tensor_shape[0], z_tensor_shape[3], -1])

	raw_score = tf.matmul(d, z) / tf.sqrt(tf.cast(d_tensor_shape[3], tf.float32))
	a_score = tf.nn.softmax(raw_score)

	return a_score

# src: [n, m, 1, d]
# attn: [n, m, m]
def Attention(h, z, filter_shape, src, name = None):
	attn = Align(h, z, filter_shape, name)

	src_tensor_shape = src.get_shape().as_list()
	#attn = tf.reshape(attn, \
	#	[attn_tensor_shape[0], attn_tensor_shape[1], 1, 1])
	src_ = tf.reshape(src, [src_tensor_shape[0], -1, src_tensor_shape[3]])

	c = tf.matmul(attn, src_)
	c_tensor_shape = c.get_shape().as_list()
	c = tf.reshape(c, [c_tensor_shape[0], -1, 1, c_tensor_shape[2]])

	return c

# input_tensor: [n, m, 1, 2] (generally)
def SeqLeftShift(input_tensor):
	tensor_shape = input_tensor.get_shape().as_list()
	dy_tensor_shape = tf.shape(input_tensor)
	new_tensor_shape = [tensor_shape[0], dy_tensor_shape[1] - 1, \
						tensor_shape[2], tensor_shape[3]]

	output_tensor = tf.slice(input_tensor, [0, 1, 0, 0], new_tensor_shape)
	return tf.pad(output_tensor, [[0, 0], [0, 1], [0, 0], [0, 0]])

def SeqRightShift(input_tensor):
	tensor_shape = input_tensor.get_shape().as_list()
	dy_tensor_shape = tf.shape(input_tensor)
	new_tensor_shape = [tensor_shape[0], dy_tensor_shape[1] - 1, \
						tensor_shape[2], tensor_shape[3]]

	output_tensor = tf.slice(input_tensor, [0, 0, 0, 0], new_tensor_shape)
	return tf.pad(output_tensor, [[0, 0], [1, 0], [0, 0], [0, 0]])

def DevConv(input_tensor, filter_shape, direction, name):
	padding = filter_shape[0] - 1

	if(direction == 'L'):
		pad_shape = [[0, 0], [padding, 0], [0, 0], [0, 0]]
	elif(direction == 'R'):
		pad_shape = [[0, 0], [0, padding], [0, 0], [0, 0]]
	else:
		print('Direction parameter error.')

	input_tensor = tf.pad(input_tensor, pad_shape)
	conv = Conv(input_tensor, filter_shape, name = name)
	return conv
