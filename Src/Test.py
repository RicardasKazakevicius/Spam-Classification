# -*- coding: utf-8 -*-
import numpy as np
from time import time
from Evaluation import evaluate_algorithm
from Dictionary import *
from Features import *
from Algorithms import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

def calculate_duration(number_of_tests, n_splits, data_dir, words_begin, words_end, words_step, standard_scale, min_max_scale, rm_stop_words, stemming, tokenization, results_dir):

	durations = []
	accurasies = []
	most_popular_words = []
	patches = []
	colors = ['#FFA07A', '#0000FF', '#8A2BE2', '#A52A2A', '#FF9912', '#3D9140', '#00CDCD', '#A9A9A9', '#FF3030', '#000000', ]
	algorithms = get_algorithms()
	dictionary, mails_count = make_dictonary(data_dir, None, rm_stop_words, stemming, tokenization)
	fig, ax = plt.subplots()
	i = 0

	for words in range(words_begin, words_end, words_step):
		dictionary_ = Counter(dictionary).most_common(words)
		print(len(dictionary), len(dictionary_))
		features, labels = extract_features_and_labels(data_dir, dictionary_, mails_count)

		start = time()
		accurasy = []
		for algo in algorithms.get():
			accurasy.append(evaluate_algorithm(algo.get(), features, labels, n_splits, standard_scale, min_max_scale))
		
		plt.plot((time()-start), np.average(accurasy), 'ro', color=colors[i])
		patches.append(mpatches.Patch(color=colors[i], label=str(words) + ' žodžių'))
		i += 1

		accurasies.append(np.average(accurasy))
		durations.append((time()-start))
		most_popular_words.append(words)

	file = open(results_dir + '.txt', 'w')
	file.write('%s\n' % accurasies)
	file.write('%s\n' % durations)
	file.write('%s\n' % most_popular_words)
	file.close()

	max_accurasy = sorted(accurasies)[-1]
	min_accurasy = sorted(accurasies)[0]
	max_duration = sorted(durations)[-1]
	min_duration = sorted(durations)[0]

	plt.legend(handles=patches)
	plt.xlabel('Trukmė sekundėmis')
	plt.ylabel('Filtravimo tikslumas procentais')
	plt.axis([min_duration-5, max_duration+5, min_accurasy-1, max_accurasy+1])
	# plt.show()
	plt.savefig(results_dir)
	plt.close()