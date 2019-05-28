from scipy.stats import ttest_ind
import numpy as np


AUTHOR_EMAIL = 'amamish@yandex.ru'

def evaluate(
	train_conversions, 
    train_indices,
    test_conversions,
    test_indices
):
	num_train = np.unique(train_indices, return_counts=True)[1]
	num_test = np.unique(test_indices, return_counts=True)[1]
	train_sum = np.zeros(500)
	test_sum = np.zeros(500)
	for i in range(10000):
		train_sum[train_indices[i]] += train_conversions[i]
		test_sum[test_indices[i]] += test_conversions[i]
	train_mean = train_sum / num_train
	test_mean = test_sum / num_test
	return ttest_ind(train_mean, test_mean, equal_var=False).pvalue
	#return np.random.uniform()
