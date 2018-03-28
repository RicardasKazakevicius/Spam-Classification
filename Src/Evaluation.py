from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def evaluate(algorithms, features, labels, number_of_tests, standard_scale=False, min_max_scale=False):

	accurasies = np.zeros((len(algorithms), number_of_tests))
	standings = np.zeros((len(algorithms), number_of_tests), dtype=int)
	
	for t in range(number_of_tests):
		print('Test: %d' % (t+1))
		train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, shuffle=True)

		if (standard_scale):
			scaler = StandardScaler()
			train_features = scaler.fit_transform(train_features)
			test_features = scaler.fit_transform(test_features)

		elif (min_max_scale):
			scaler = MinMaxScaler(copy=False)
			train_features = scaler.fit_transform(train_features)
			test_features = scaler.fit_transform(test_features)

		# Training all algorithms
		for a in range(len(algorithms)):
			algorithms[a]['Algo'].fit(train_features, train_labels)
			
		# Predicting and saving accurasies to array for each algorithm in each test
		for a in range(len(algorithms)):
			prediction = algorithms[a]['Algo'].predict(test_features)
			accurasies[a][t] = accuracy_score(test_labels, prediction)*100
			
	
	# Saving accurasies array to each algorithm
	for a in range(len(algorithms)):
			algorithms[a]['Accuracy'] = accurasies[a]


	# Counting how many times each algorithm took certain position comparing with rest algorithms accurasies
	for t in range(number_of_tests):	
		for i in range(len(algorithms)):
			position = 1
			for j in range(len(algorithms)):
				if(algorithms[i]['Accuracy'][t] < algorithms[j]['Accuracy'][t]):
					position += 1
			
			standings[i][t] = position

	# Saving standings to each algorithm
	for i in range(len(algorithms)):
		algorithms[i]['Standings'] = standings[i]

	return algorithms

