import tensorflow as tf

# pred: tensor of prediction
# grd: tensor of ground truth
def PredictionResult(pred, grd):
	prediction = tf.argmax(pred, 1)
	ground_truth = tf.argmax(grd, 1)
	correctness = tf.equal(prediction, ground_truth)
	accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
	return prediction, correctness, accuracy

