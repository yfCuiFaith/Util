import time

def CurrentTime():
	return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def ReadDataSet(data_dir, feature_file, label_file, dataset_type, dataset_name = None):
	if(dataset_name != None):
		print('Reading %s dataset...' % (dataset_name))

	dataset = dataset_type(data_dir + feature_file, \
								data_dir + label_file)

	if(dataset_name != None):
		print('Reading %s dataset complete.' % (dataset_name))

	return dataset
