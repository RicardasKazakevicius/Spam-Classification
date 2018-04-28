from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools as it
import random
from time import time
import operator

def evaluate_algorithms(algorithms, features, labels, n_repeats, n_splits, standard_scale, min_max_scale):
	
	test = 1
	rkf = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits)

	for train_index, test_index in rkf.split(features):
		# print test, 
		train_features, test_features = features[train_index], features[test_index]
		train_labels, test_labels = labels[train_index], labels[test_index]

		if (standard_scale):
			train_features, test_features = scale_standard(train_features, test_features)

		elif (min_max_scale):
			train_features, test_features = scale_min_max(train_features, test_features)			
		
		algorithms.fit(train_features, train_labels)		
		algorithms.predict(test_features, test_labels)
		algorithms.set_standings()

		test+=1
	return algorithms


def get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale):

	algo_start = time()
	results = []
	
	combinations, params_names = get_all_combinations(param_grids)
	
	for combination in combinations:	
		params = dict(zip(params_names, combination))
		algorithm.set_params(**params)		
		accurasy = evaluate_algorithm(algorithm, features, labels, n_splits, standard_scale, min_max_scale)
		results.append((params, accurasy))
	print(str(algorithm).split('(')[0], (time()-algo_start)/60)

	best_result = sorted(results, key=operator.itemgetter(1))[-1]
	algorithm.set_params(**best_result[0])

	print(algorithm)
	print(best_result[1])
	return algorithm


def get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	algo_start = time()
	results = []

	combinations, params_names = get_all_combinations(param_grids)	
	random_combinations = get_random_combinations(combinations, n_iter)

	for combination in random_combinations:	
		params = dict(zip(params_names, combination))
		algorithm.set_params(**params)		
		accurasy = evaluate_algorithm(algorithm, features, labels, n_splits, standard_scale, min_max_scale)
		results.append((params, accurasy))
	print(str(algorithm).split('(')[0], (time()-algo_start)/60)

	best_result = sorted(results, key=operator.itemgetter(1))[-1]
	algorithm.set_params(**best_result[0])
	print(algorithm)
	print(best_result[1])
	return algorithm


def evaluate_algorithm(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	accurasies = 0
	kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)

	for train_index, test_index in kf.split(features):
		train_features, test_features = features[train_index], features[test_index]
		train_labels, test_labels = labels[train_index], labels[test_index]
		
		if (standard_scale):
			train_features, test_features = scale_standard(train_features, test_features)

		elif (min_max_scale):
			train_features, test_features = scale_min_max(train_features, test_features)	
		
		algorithm.fit(train_features, train_labels)		
		prediction = algorithm.predict(test_features)
		accurasies += accuracy_score(test_labels, prediction)*100

	return accurasies/n_splits


def get_all_combinations(param_grids):

	combinations = []
	all_keys = []
	for param_grid in param_grids:
		all_keys = sorted(param_grid)
		combinations += list(it.product(*(param_grid[key] for key in all_keys)))
	return combinations, all_keys


def get_random_combinations(combinations, amount):

	random_combinations = []
	for i in range(amount):
		combination = random.choice(combinations)
		combinations.remove(combination)
		random_combinations.append(combination)
	return random_combinations


def scale_standard(train_features, test_features):
	scaler = StandardScaler()
	train_features = scaler.fit_transform(train_features)
	test_features = scaler.fit_transform(test_features)
	return train_features, test_features


def scale_min_max(train_features, test_features):
	scaler = MinMaxScaler(copy=False)
	train_features = scaler.fit_transform(train_features)
	test_features = scaler.fit_transform(test_features)
	return train_features, test_features