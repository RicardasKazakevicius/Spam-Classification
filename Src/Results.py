from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import os

def save_info(algorithms, features, labels, directory, time):
	
	if not os.path.exists(directory):
		os.makedirs(directory)
	file = open(directory + 'results.txt', 'w')
	file.write('%f minutes\n' % (time/60))
	file.write('%s:\n' % directory.split('/')[-2])
	file.write('Emails: %d\n' % features.shape[0])
	file.write('Words: %d\n' % features.shape[1])
	file.write('Ham: %d\n' % sum(labels==0))
	file.write('Spam: %d\n' % sum(labels==1))

	file.write('\n')
	for i in range(len(algorithms)):
		file.write(str(algorithms[i]['Name']) + '\n') 
		file.write(str(algorithms[i]['Accuracy']) + '\n')
		file.write('{:.2f}'.format(np.average(algorithms[i]['Accuracy'])) + '\n')
		file.write('{:.2f}'.format(np.std(algorithms[i]['Accuracy'])) + '\n')
		file.write(str(algorithms[i]['Standings']) + '\n')

	if (len(algorithms[0]) > 4):
		file.write('\n')	
		for i in range(len(algorithms)):
			file.write(str(algorithms[i]['Name']) + '\n') 
			file.write('Number of parameters combination: %d\n' % np.prod(algorithms[i]['Params_info']))

	file.write('\n')
	for i in range(len(algorithms)):
		file.write('%s\n' % algorithms[i]['Algo'].get_params)

	file.close()

	if not os.path.exists(directory):
		os.makedirs(directory)
	file = open(directory + 'accurasies.txt', 'w')
	for i in range(len(algorithms)):
		file.write('{:.2f}'.format(np.average(algorithms[i]['Accuracy'])) + '\n')
	file.close()

	if not os.path.exists(directory):
		os.makedirs(directory)
	file = open(directory + 'stds.txt', 'w')
	for i in range(len(algorithms)):
		file.write('{:.2f}'.format(np.std(algorithms[i]['Accuracy'])) + '\n')
	file.close()

def save_plots(algorithms, directory, number_of_tests):

	for i in range(len(algorithms)):

		standing = dict(Counter(algorithms[i]['Standings']))
		empty = {}
		for j in range(1,len(algorithms)+1):
			empty.update({j: 0})

		std = {k: standing.get(k, 0) + empty.get(k, 0) for k in set(standing) | set(empty)}

		plt.title(algorithms[i]['Name'], fontsize=17)
		plt.xlabel('Vieta', fontsize=17)
		plt.ylabel('Kartai', fontsize=17)
		rectanges = plt.bar(range(len(std)), list(std.values()), align='center')
		plt.xticks(range(len(std)), list(std.keys()))
		plt.yticks(np.arange(0, number_of_tests+10, 10))
		
		for rect in rectanges:
			if(rect.get_height() > 0):
				plt.text(rect.get_x() + rect.get_width()/2, rect.get_height(), 
								 rect.get_height(), ha='center', va='bottom')
		plt.savefig(directory + str(i))
		plt.close()



	