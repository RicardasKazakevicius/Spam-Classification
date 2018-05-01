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
	
	# data_dir = '../Spam-Assasin'
	data_dir = '../Ling-Spam'
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
	
	number_of_tests = 4

	# Number of most common words
	vector_size = 100

	dictionary, mails_count = make_dictonary(data_dir, vector_size, rm_stop_words, stemming, tokenization)
	features, labels = extract_features_and_labels(data_dir, dictionary, mails_count)
	
	# # np.save(data_dir[3:] + '_' + str(vector_size) + '_features', features)
	# # np.save(data_dir[3:] + '_' + str(vector_size) + '_labels' , labels)
	
	# # features =  np.load(data_dir[3:] + '_' + str(vector_size) + '_features.npy')
	# # labels = np.load(data_dir[3:] + '_' + str(vector_size) + '_labels.npy')

	# algorithms = get_algorithms()
	# results_dir = results_dir + data_dir[3:] + '_' + str(vector_size) + 'temp1/'

	# # algorithms = get_grid_search_tuned_algorithms(features, labels, n_splits, standard_scale, min_max_scale)
	# # results_dir = results_dir + data_dir[3:] + '_' + str(vector_size) + '_GridSearch/'

	# # algorithms = get_random_search_tuned_algorithms(features, labels, n_splits, n_iter, standard_scale, min_max_scale)
	# # results_dir = results_dir + data_dir[3:] + '_' + str(vector_size) + '_RandomSearch/'

	# algorithms = evaluate_algorithms(algorithms, features, labels, number_of_tests/n_splits, n_splits, standard_scale, min_max_scale)

	# save_test_results(algorithms, features, labels, results_dir, time()-start)
	# create_plots(algorithms, results_dir, number_of_tests)


	# Test for finding opimal vector size
	# words_begin = 100
	# words_end = 1100
	# words_step = 100
	# results_dir = results_dir + data_dir[3:] + '_DurationAccurasyRatio'
	# vector_duration_accurasy_ratio(number_of_tests, n_splits, data_dir, words_begin, words_end, words_step, standard_scale, min_max_scale, rm_stop_words, stemming, tokenization, results_dir)


