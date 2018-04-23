import math

# pred: prediction, 0 or 1
# rw: right or wrong, bool
def Statistic(pred, rw):
	if(len(pred) != len(rw)):
		return None
	
	tp = 0.
	tn = 0.
	fp = 0.
	fn = 0.
	
	for i in range(0, len(rw)):
		if(rw[i]):
			if(pred[i] == 1):
				tp += 1
			else:
				tn += 1
		else:
			if(pred[i] == 1):
				fp += 1
			else:
				fn += 1
	return tp, tn, fp, fn

def GetMCC(tp, tn, fp, fn):
	deno = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
	if(deno):
		return (tp * tn - fp * fn) / deno
	else:
		return 0.0

def Assess(pred, rw):
	TP, TN, FP, FN = Statistic(pred, rw)
	print('TP = %6d TN = %6d FP = %6d FN = %6d' % (TP, TN, FP, FN))
	
	precision = TP / (TP + FP + 0.001)
	sensitivity = TP / (TP + FN + 0.001)
	specificity = TN / (TN + FP + 0.001)
	print('Precision = %.2f%% Sensitivity = %.2f%% Specificity = %.2f%%' % \
		(precision * 100.0, sensitivity * 100.0, specificity * 100.0))
	
	MCC = GetMCC(TP, TN, FP, FN)
	print('MCC = %.3f' % (MCC))
