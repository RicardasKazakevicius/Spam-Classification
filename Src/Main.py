from time import time
from Algorithms import *
from Dictionary import *
from Features import *
from Results import *
from Evaluation import *
from VectorSizeTest import *
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
	
	start = time()
	
	results_dir = '../Results/'

	data_dir = '../Spam-Assasin'
	# data_dir = '../Ling-Spam'
	# data_dir = '../PU1'
	# data_dir = '../Enron-small'

	# number of k_fold cross-validation
	n_splits = 4

	# number of iterations in random parameters search
	n_iter = 96

	# Scale features, making features values with mean zero and standard deviation one
	standard_scale = False

	# Scale features, making features values between zero and one
	min_max_scale = True

	rm_stop_words = True
	stemming = True
	tokenization = True
	
	number_of_tests = 100

	# Number of most common words
	vector_size = 600

	# Extracting features and labels data
	# dictionary, mails_count = make_dictonary(data_dir, vector_size, rm_stop_words, stemming, tokenization)
	# features, labels = extract_features_and_labels(data_dir, dictionary, mails_count)
	
	# Saving features and labels to files
	# np.save(data_dir[3:] + '_' + str(vector_size) + '_MinMax_features', features)
	# np.save(data_dir[3:] + '_' + str(vector_size) + '_MinMax_labels' , labels)
	
	# Reading features and labels from files
	features =  np.load(data_dir[3:] + '_' + str(vector_size) + '_MinMax_features.npy')
	labels = np.load(data_dir[3:] + '_' + str(vector_size) + '_MinMax_labels.npy')

	# algorithms = get_algorithms()
	# results_dir = results_dir + data_dir[3:] + '_' + str(vector_size) + '/'

	# algorithms = get_grid_search_tuned_algorithms(features, labels, n_splits, standard_scale, min_max_scale)
	# results_dir = results_dir + data_dir[3:] + '_' + str(vector_size) + '_GridSearch/'

	algorithms = get_random_search_tuned_algorithms(features, labels, n_splits, n_iter, standard_scale, min_max_scale)
	results_dir = results_dir + data_dir[3:] + '_' + str(vector_size) + '_RandomSearch/'

	algorithms = evaluate_algorithms(algorithms, features, labels, number_of_tests, n_splits, standard_scale, min_max_scale)

	save_test_results(algorithms, features, labels, results_dir, time()-start)


	# Finding opimal vector size
	# size_begin = 100
	# size_end = 1100
	# size_step = 100
	# results_dir = results_dir + data_dir[3:] + '_DurationAccurasyRatio'
	# vector_duration_accurasy_ratio(number_of_tests, n_splits, data_dir, size_begin, size_end, size_step, standard_scale, min_max_scale, rm_stop_words, stemming, tokenization, results_dir)


