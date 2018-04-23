import sys
import os
sys.path.append(os.getcwd())
from test import Assess

# dict_key: (feature_tensor, label_tensor, keep_prob, is_test)
def Valid(sess, model, dict_key, dataset, batch_size = None):
	if(batch_size is None):
		feature, label = dataset.EntireBatch()
	else:
		feature, label = dataset.NextBatch(batch_size)

	input_dict = {}

	input_dict[dict_key[0]] = feature
	input_dict[dict_key[1]] = label
	input_dict[dict_key[2]] = 1.0
	input_dict[dict_key[3]] = True

	val_pred, val_rw, val_acc = \
		sess.run(model, feed_dict = input_dict)

	print('ValAccuracy = %.2f%%' % (val_acc * 100))
	
	Assess(val_pred, val_rw)

def ValidWithBatch(sess, model, dict_key, dataset, batch_size):
	total_pred = []
	total_rw = []
	total_acc = 0.0
	total_iter = 0

	dataset.SetSignal()
	
	while(True):
		feature, label = dataset.NextRestrictedBatch(batch_size)

		if(label is None):
			break

		input_dict = {}

		input_dict[dict_key[0]] = feature
		input_dict[dict_key[1]] = label
		input_dict[dict_key[2]] = 1.0
		input_dict[dict_key[3]] = True

		val_pred, val_rw, val_acc = \
			sess.run(model, feed_dict = input_dict)
		
		total_pred.extend(val_pred)
		total_rw.extend(val_rw)
		total_acc += val_acc
		total_iter += 1
		
	print ('ValAccuracy = %.2f%%' % (total_acc * 100 / total_iter))
	
	Assess(total_pred, total_rw)

def Execute(sess, train_model, analy_model, \
			fetch_list, dict_key, \
			train_dataset, tr_batch_size, \
			val_dataset, v_batch_size, \
			begin, end, lr = None):
	fetch_list = [train_model] + fetch_list + [analy_model]
	# [train_model, train_loss, cross_entropy, L2, learning_rate, analy_model]
	
	for i in range(begin, end):
		batch_x, batch_y = train_dataset.NextBatch(tr_batch_size)

		input_dict = {}

		input_dict[dict_key[0]] = batch_x
		input_dict[dict_key[1]] = batch_y
		input_dict[dict_key[2]] = 0.5
		input_dict[dict_key[3]] = False

		if(lr):
			input_dict[dict_key[4]] = lr

		if(i % 100):
			buf, loss, ce, L2, cur_lr = \
				sess.run(fetch_list[:-1], feed_dict = input_dict)
		else:
			buf, loss, ce, L2, cur_lr, (pred, rw, acc) = \
				sess.run(fetch_list, feed_dict = input_dict)

			print('Iter %d\nTrainAccuracy = %.2f%% lr = %.8f loss = %.6f ce = %.6f L2 = %.6f' \
				% (i, acc * 100, cur_lr, loss, ce, L2))
			Assess(pred, rw)

			if(i % 1000):
				continue

			Valid(sess, analy_model, dict_key[:-1], val_dataset, batch_size = v_batch_size)
	pass

