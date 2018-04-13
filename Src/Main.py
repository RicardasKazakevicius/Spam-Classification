from time import time
from Algorithms import *
from Dictionary import *
from Features import *
from Results import *
from Evaluation import *
from sklearn.neural_network import MLPClassifier
if __name__ == '__main__':

	start = time()
	
	results_dir = '../Results/'
	
	# data_dir = '../All_Data'
	# data_dir = '../Enron'
	# data_dir = '../Spam-Assasin'
	# data_dir = '../Ling-Spam'
	data_dir = '../PU1'
	# data_dir = '../Ling-Assasin-Enron'
	# data_dir = '../Enron-small'

	number_of_tests = 1
	most_common_words = 500

	# dictionary, mails_count = make_dictonary(data_dir, most_common_words, rm_stop_words=True, stemming=True)
	# features, labels = extract_features_and_labels(data_dir, dictionary, mails_count, False, False)
	# np.save(data_dir[3:] + '_' + str(most_common_words) + '_features', features)
	# np.save(data_dir[3:] + '_' + str(most_common_words) + '_labels' , labels)
	
	features =  np.load(data_dir[3:] + '_' + str(most_common_words) + '_features.npy')
	labels = np.load(data_dir[3:] + '_' + str(most_common_words) + '_labels.npy')

	# algorithms = get_algorithms()
	algorithms = get_tuned_algorithms(features, labels, n_jobs=1, verbose=1, n_iter=1, cv=5)

	algorithms = evaluate(algorithms, features, labels, number_of_tests, False, False)

	# results_dir = results_dir + data_dir[3:] + '_' + str(most_common_words) + '/'
	# results_dir = results_dir + data_dir[3:] + '_' + str(most_common_words) + '_Tuned/'
	results_dir = results_dir + data_dir[3:] + '_' + str(most_common_words) + '_Temp/'
	save_info(algorithms, features, labels, results_dir, time()-start)
	save_plots(algorithms, results_dir, number_of_tests)