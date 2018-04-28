from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import os

def save_test_results(algorithms, features, labels, directory, time):
	
	if not os.path.exists(directory):
		os.makedirs(directory)
	file = open(directory + 'results.txt', 'w')
	file.write('%f minutes\n' % (time/60))
	file.write('%s:\n' % directory.split('/')[-2])
	file.write('Emails: %d\n' % features.shape[0])
	file.write('Words: %d\n' % features.shape[1])
	file.write('Ham: %d\n' % sum(labels==0))
	file.write('Spam: %d\n' % sum(labels==1))

	file.write('\nAverage accurasy:\n')
	for algo in algorithms.get():
		file.write('{:25s}'.format(algo.get_name()))
		file.write('{:.2f}\n'.format(np.average(algo.get_accurasies())))

	file.write('\nStandard deviation:\n')
	for algo in algorithms.get():
		file.write('{:25s}'.format(algo.get_name()))
		file.write('{:.2f}\n'.format(np.std(algo.get_accurasies())))

	file.write('\nAccurasies:\n')
	for algo in algorithms.get():
		file.write('{:25s}'.format(algo.get_name())) 
		file.write('%s\n' % algo.get_accurasies())

	file.write('\nStandings:\n')
	for algo in algorithms.get():
		file.write('{:25s}'.format(algo.get_name()))
		file.write('%s\n' % algo.get_standings())

	file.write('\nParameters:\n')
	for algo in algorithms.get():
		file.write('%s:\n' % algo.get_name())
		file.write('%s\n' % algo.get().get_params)
	file.close()


def create_plots(algorithms, directory, number_of_tests):

	for algo in algorithms.get():
		standings = dict(Counter(algo.get_standings()))

		empty = {}
		for j in range(1,len(algorithms.get())+1):
			empty.update({j: 0})

		std = {k: standings.get(k, 0) + empty.get(k, 0) for k in set(standings) | set(empty)}

		plt.title(algo.get_name(), fontsize=17)
		plt.xlabel('Vieta', fontsize=17)
		plt.ylabel('Kartai', fontsize=17)
		rectanges = plt.bar(range(len(std)), list(std.values()), align='center')
		plt.xticks(range(len(std)), list(std.keys()))
		plt.yticks(np.arange(0, number_of_tests+10, 10))
		
		for rect in rectanges:
			if(rect.get_height() > 0):
				plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), 
								 rect.get_height(), ha='center', va='bottom')
		plt.savefig(directory + algo.get_name())
		plt.close()