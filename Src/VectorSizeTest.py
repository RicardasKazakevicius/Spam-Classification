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
import pandas as pd
reload(sys)  
sys.setdefaultencoding('utf8')

def vector_duration_accurasy_ratio(number_of_tests, n_splits, data_dir, words_begin, words_end, words_step, standard_scale, min_max_scale, rm_stop_words, stemming, tokenization, results_dir):

	durations = []
	accurasies = []
	vector_sizes = []
	patches = []
	colors = ['#FFA07A', '#0000FF', '#8A2BE2', '#A52A2A', '#FF9912', '#3D9140', '#00CDCD', '#A9A9A9', '#FF3030', '#000000', ]
	algorithms = get_algorithms()
	dictionary, mails_count = make_dictonary(data_dir, None, rm_stop_words, stemming, tokenization)
	i = 0

	for words in range(words_begin, words_end, words_step):
		dictionary_ = Counter(dictionary).most_common(words)
		print(len(dictionary_))
		features, labels = extract_features_and_labels(data_dir, dictionary_, mails_count)

		start = time()
		accuracy = []
		for algo in algorithms.get():
			accuracy.append(evaluate_algorithm(algo.get(), features, labels, n_splits, standard_scale, min_max_scale))
		
		plt.plot((time()-start), np.average(accuracy), 'ro', color=colors[i], label=str(words))
		i += 1

		accurasies.append(round(np.average(accuracy), 2))
		durations.append(round((time()-start), 2))
		vector_sizes.append(words)

	accuracy_difference = []
	duration_difference = []
	difference = []
	for i in range(len(accurasies)-1):
		accuracy_difference.append(round(accurasies[i+1] - accurasies[i], 2))
		duration_difference.append(round(durations[i+1] - durations[i], 2))
		difference.append(round(((accurasies[i+1] - accurasies[i]) / (durations[i+1] - durations[i])), 3))

	max_accurasy = sorted(accurasies)[-1]
	min_accurasy = sorted(accurasies)[0]
	max_duration = sorted(durations)[-1]
	min_duration = sorted(durations)[0]

	plt.legend(title='Vektoriaus dydis', bbox_to_anchor=(0.72, 0.66), loc=2)
	plt.xlabel('Trukmė sekundėmis')
	plt.ylabel('Filtravimo tikslumas procentais')
	plt.axis([min_duration-5, max_duration+5, min_accurasy-1, max_accurasy+1])
	plt.savefig(results_dir)
	plt.close()

	file = open(results_dir + '.txt', 'w')
	file.write('%s\n%s\n%s\n%s\n%s\n%s' % 
		(vector_sizes, accurasies, durations, accuracy_difference, duration_difference, difference))
	file.close()