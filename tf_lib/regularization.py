import tensorflow as tf

def L2Loss(param_list):
	l2_loss = tf.nn.l2_loss(param_list[0])
	for i in range(1, len(param_list)):
		l2_loss += tf.nn.l2_loss(param_list[i])
	return l2_loss
