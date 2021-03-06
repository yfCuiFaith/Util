import numpy as np
import gc

class DataSet:
	def ReadDataSource(self, src):
		if(isinstance(src, np.ndarray)):
			return src
		
		with open(src) as srcFile:
			buf = srcFile.readlines()
		
		for i in range(0, len(buf)):
			buf[i] = buf[i].strip().split(' ')
			for j in range(0, len(buf[i])):
				buf[i][j] = float(buf[i][j])
		res = np.array(buf)
		
		del buf
		gc.collect()
		
		return res
	
	def __init__(self, featureSource = None, labelSource = None):
		self.index = 0
		self.number = 0
		self.feature_ndarray = None
		self.label_ndarray = None
		
		if(featureSource is not None):
			self.feature_ndarray = self.ReadDataSource(featureSource)
			self.number = len(self.feature_ndarray)
		if(labelSource is not None):
			self.label_ndarray = self.ReadDataSource(labelSource)
			if(self.number == 0):
				self.number = len(self.label_ndarray)
			elif(self.number != len(self.label_ndarray)):
				print '###DataError!###'
			pass
		pass
	
	def NextBatch(self, batch_size):
		feature_batch = None
		label_batch = None
		
		if(self.index + batch_size <= self.number):
			feature_batch = \
				self.feature_ndarray[self.index:self.index + batch_size]
			label_batch = \
				self.label_ndarray[self.index:self.index + batch_size]
			self.index = self.index + batch_size
			
			if(self.index == self.number):
				self.index = 0
			
			return feature_batch.tolist(), label_batch.tolist()
		
		next_end = (self.index + batch_size) % self.number
		
		feature_batch = np.row_stack((self.feature_ndarray[self.index:], \
										self.feature_ndarray[:next_end]))
		label_batch = np.row_stack((self.label_ndarray[self.index:], \
										self.label_ndarray[:next_end]))
		self.index = next_end
		
		return feature_batch.tolist(), label_batch.tolist()
	
	def EntireBatch(self):
		return self.feature_ndarray.tolist(), self.label_ndarray.tolist()

class AutoBalDataSet:
	def ReadDataSource(self, src):
		if(isinstance(src, np.ndarray)):
			return src
		
		with open(src) as srcFile:
			buf = srcFile.readlines()
		
		for i in range(len(buf)):
			buf[i] = buf[i].strip().split()
		
		res = np.array(buf, dtype = np.float32)
		
		del buf
		gc.collect()
		
		return res
	
	def __init__(self, featureSource = None, labelSource = None):
		self.posi_number = 0
		self.nega_number = 0
		
		buf_size = 1000000
		
		feature_list = None
		
		label_ndarray = self.ReadDataSource(labelSource)
		
		posi_std = np.array([0, 1], dtype = np.float32)
		nega_std = np.array([1, 0], dtype = np.float32)
		
		posi_index = np.unique(np.where(label_ndarray == posi_std)[0])
		nega_index = np.unique(np.where(label_ndarray == nega_std)[0])
		
		posi_label = label_ndarray[posi_index]
		nega_label = label_ndarray[nega_index]
		
		featureReader = open(featureSource)
		
		dim = len(featureReader.readline().strip().split())
		featureReader.seek(0)
		
		posi_feature = np.empty(\
			shape = [len(posi_label), dim], dtype = np.float32)
		nega_feature = np.empty(\
			shape = [len(nega_label), dim], dtype = np.float32)
		
		posi_number = 0
		nega_number = 0
		index = 0
		pre = ''
		cur = ''
		
		gc.collect()
		
		while(True):
			buf = featureReader.read(buf_size)
			
			if(buf == ''):
				break

			rindex_ent = buf.rindex('\n')
			cur = pre + buf[:rindex_ent]
			pre = buf[rindex_ent + 1:] # rindex_ent may be no less than len(buf)
			
			cur = cur.strip().split('\n')
			for i in range(len(cur)):
				cur[i] = cur[i].split()
			buf = np.array(cur, dtype = np.float32)
			
			seg = label_ndarray[index: index + len(buf)]
			index += len(buf)
			posi_index = np.unique(np.where(seg == posi_std)[0])
			nega_index = np.unique(np.where(seg == nega_std)[0])
			
			posi_feature[posi_number: posi_number + len(posi_index)] = \
				buf[posi_index]
			nega_feature[nega_number: nega_number + len(nega_index)] = \
				buf[nega_index]
			
			posi_number += len(posi_index)
			nega_number += len(nega_index)
		
		featureReader.close()

		del label_ndarray
		del posi_index, nega_index
		gc.collect()
		
		self.p_rate_coef = 0.5
		self.posi_number = posi_number
		self.nega_number = nega_number
		
		self.posi_dataset = DataSet(posi_feature, posi_label)
		self.nega_dataset = DataSet(nega_feature, nega_label)
	
	def SetPositiveRateCoef(self, value):
		self.p_rate_coef = value

	def NextBatch(self, batch_size):
		posi_batch_size = int(batch_size * self.p_rate_coef)
		nega_batch_size = batch_size - posi_batch_size
		
		posi_batch = self.posi_dataset.NextBatch(posi_batch_size)
		nega_batch = self.nega_dataset.NextBatch(nega_batch_size)
		
		return posi_batch[0] + nega_batch[0], posi_batch[1] + nega_batch[1]

	def SetSignal(self):
		self.posi_signal = self.posi_number
		self.nega_signal = self.nega_number
	
	def NextRestrictedBatch(self, batch_size):
		if(self.posi_signal > 0):
			batch_size = min(batch_size, self.posi_signal)
			self.posi_signal -= batch_size
			return self.posi_dataset.NextBatch(batch_size)

		if(self.nega_signal > 0):
			batch_size = min(batch_size, self.nega_signal)
			self.nega_signal -= batch_size
			return self.nega_dataset.NextBatch(batch_size)

		return [None, None]

	def EntireBatch(self):
		posi_batch = self.posi_dataset.EntireBatch()
		nega_batch = self.nega_dataset.EntireBatch()
		return posi_batch[0] + nega_batch[0], posi_batch[1] + nega_batch[1]

# list only
class SeqDataSet:
	def ReadFeatureSource(self, src):
		if(isinstance(src, list)):
			return src

		with open(src) as srcFile:
			buf = srcFile.readlines()

		for i in range(len(buf)):
			buf[i] = buf[i].strip().split(' ')
			for j in range(len(buf[i])):
				buf[i][j] = float(buf[i][j])

		return buf

	def ReadLabelSource(self, src):
		if(isinstance(src, list)):
			return src

		with open(src) as srcFile:
			buf = srcFile.readlines()

		for i in range(len(buf)):
			buf[i] = list(buf[i].strip())
			for j in range(len(buf[i])):
				if(buf[i][j] == '0'):
					buf[i][j] = [1.0, 0.0]
				else:
					buf[i][j] = [0.0, 1.0] 
		
		return buf

	def __init__(self, featureSource, labelSource):
		self.feature = self.ReadFeatureSource(featureSource)
		self.label = self.ReadLabelSource(labelSource)
		self.index = 0

		if(len(self.feature) != len(self.label)):
			self.number = -1
			print('###DataError!###')
		else:
			self.number = len(self.label)

	def Number(self):
		return self.number

	def Lengths(self):
		return [len(seq) for seq in self.label]
	
	def Fill(self, src_seqs, filler):
		seqs_lens = [len(seq) for seq in src_seqs]

		batch_size = len(src_seqs)

		max_len = max(seqs_lens)

		dest_seqs = []
		for i in range(0, batch_size):
			pad_size = max_len - seqs_lens[i]
			dest_seqs.append(src_seqs[i] + \
				[filler for j in range(pad_size)])
		
		return dest_seqs, seqs_lens

	def NextBatch(self, batch_size = 1):
		feature_batch = None
		label_batch = None
		
		if(self.index + batch_size <= self.number):
			feature_batch = \
				self.feature[self.index: self.index + batch_size]
			label_batch = \
				self.label[self.index: self.index + batch_size]
			self.index = self.index + batch_size
			
			if(self.index == self.number):
				self.index = 0
			
			return feature_batch, label_batch
		
		next_end = (self.index + batch_size) % self.number
		
		feature_batch = \
			self.feature[self.index:] + self.feature[:next_end]
		label_batch = \
			self.label[self.index:] + self.label[:next_end]
		self.index = next_end
		
		return feature_batch, label_batch

	def NextPaddingBatch(self, batch_size = 1):
		batch_x, batch_y = self.NextBatch(batch_size)

		batch_x, len_x = self.Fill(batch_x, 0.0)
		batch_y, len_y = self.Fill(batch_y, [0.0, 0.0])

		return batch_x, batch_y, len_x, len_y

	def SetSignal(self):
		self.avail_num = self.number

	# |dataset| % batch_size must be 0
	def NextRestrictedBatch(self, batch_size = 1):
		if(self.avail_num > 0):
			self.avail_num -= batch_size
			return self.NextBatch(batch_size)
		return [None, None]

	def NextRestrictedPaddingBatch(self, batch_size = 1):
		if(self.avail_num > 0):
			batch_size = min(self.avail_num, batch_size)
			self.avail_num -= batch_size
			return self.NextPaddingBatch(batch_size)
		return [None, None, None, None]

	def EntireBatch(self):
		return self.feature, self.label

class HistoryPool:
	def __init__(self, init_lengths):
		self.index = 0
		self.pool = []
		self.lengths = init_lengths
		self.size = len(init_lengths)

		for i in range(len(init_lengths)):
			self.pool.append([[[0.0, 0.0]] for j in range(init_lengths[i])])
	
	def Forget(self):
		for i in range(len(self.pool)):
			self.pool[i] = [[[0.0, 0.0]] for j in range(len(self.pool[i]))]

	def Update(self, update_tensor):
		batch_size = len(update_tensor)
		for i in range(0, batch_size):
			self.pool[self.index] = update_tensor[i][:self.lengths[self.index]]
			self.index = (self.index + 1) % self.size

	def Get(self, batch_size, filler):
		lengths = [len(self.pool[i % self.size]) \
			for i in range(self.index, self.index + batch_size)]
		max_len = max(lengths)

		res = []
		for i in range(batch_size):
			pad_size = max_len - lengths[i]

			prim_tensor = self.pool[(self.index + i) % self.size]
			padding = [filler for j in range(pad_size)]

			if(isinstance(prim_tensor, np.ndarray)):
				prim_tensor = prim_tensor.tolist()

			res.append(prim_tensor + padding)

		return res

	def GoAhead(self, batch_size):
		self.index = (self.index + batch_size) % self.size
