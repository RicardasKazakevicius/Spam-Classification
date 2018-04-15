from time import time
from Algorithms import *
from Dictionary import *
from Features import *
from Results import *
from Evaluation import *
from sklearn.neural_network import MLPClassifier
if __name__ == '__main__':

	start = time()
	
	results_dir = '../ResultsMinMax/'
	
	# data_dir = '../All_Data'
	# data_dir = '../Enron'
	# data_dir = '../Spam-Assasin'
	# data_dir = '../Ling-Spam'
	data_dir = '../PU1'
	# data_dir = '../Ling-Assasin-Enron'
	# data_dir = '../Enron-small'

	# number of k_fold cross-validation
	n_splits = 4

	# number of iterations in random parameters search
	n_iter = 96

	# Scale features with mean zero and standard deviation one
	standard_scale = False

	# Scale features with values between zero and one
	min_max_scale = True
	
	number_of_tests = 100
	most_common_words = 500

	# dictionary, mails_count = make_dictonary(data_dir, most_common_words, rm_stop_words=True, stemming=True)
	# features, labels = extract_features_and_labels(data_dir, dictionary, mails_count)
	# np.save(data_dir[3:] + '_' + str(most_common_words) + '_features', features)
	# np.save(data_dir[3:] + '_' + str(most_common_words) + '_labels' , labels)
	
	features =  np.load(data_dir[3:] + '_' + str(most_common_words) + '_features.npy')
	labels = np.load(data_dir[3:] + '_' + str(most_common_words) + '_labels.npy')

	# algorithms = get_algorithms()
	# algorithms = get_grid_search_tuned_algorithms(features, labels, n_splits, standard_scale, min_max_scale)
	algorithms = get_random_search_tuned_algorithms(features, labels, n_splits, n_iter, standard_scale, min_max_scale)
	

	# algorithms = evaluate_algorithms(algorithms, features, labels, number_of_tests/n_splits, n_splits, standard_scale, min_max_scale)

	# # results_dir = results_dir + data_dir[3:] + '_' + str(most_common_words) + '/'
	# results_dir = results_dir + data_dir[3:] + '_' + str(most_common_words) + '_GridSearch/'
	# # results_dir = results_dir + data_dir[3:] + '_' + str(most_common_words) + '_RandomSearch/'
	# save_info(algorithms, features, labels, results_dir, time()-start)
	# save_plots(algorithms, results_dir, number_of_tests)