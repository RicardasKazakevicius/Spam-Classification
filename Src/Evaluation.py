from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools as it
import random
from time import time

def evaluate_algorithms(algorithms, features, labels, n_repeats, n_splits, standard_scale, min_max_scale):
	
	rkf = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=42)

	for train_index, test_index in rkf.split(features):
		train_features, test_features = features[train_index], features[test_index]
		train_labels, test_labels = labels[train_index], labels[test_index]

		if (standard_scale):
			train_features, test_features = scale_standard(train_features, test_features)

		elif (min_max_scale):
			train_features, test_features = scale_min_max(train_features, test_features)			
		
		algorithms.fit(train_features, train_labels)		
		algorithms.predict(test_features, test_labels)
		algorithms.set_standings()

	return algorithms


def get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale):

	start = time()
	best_accurasy = 0
	for param_grid in param_grids:
		all_keys = sorted(param_grid)
		combinations = list(it.product(*(param_grid[key] for key in all_keys)))

		for combination in combinations:	
			params = dict(zip(all_keys, combination))
			algorithm.set_params(**params)		
			accurasy = evaluate_algorithm(algorithm, features, labels, n_splits, standard_scale, min_max_scale)
			if (accurasy > best_accurasy):
				best_accurasy = accurasy
				best_algorithm = algorithm
		print(best_algorithm, len(combinations), best_accurasy, (time()-start)/60)
	return best_algorithm


def get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	start = time()
	best_accurasy = 0
	for param_grid in param_grids:
		all_keys = sorted(param_grid)
		combinations = list(it.product(*(param_grid[key] for key in all_keys)))
		random_combinations = []

		for i in range(n_iter/len(param_grids)):
			combination = random.choice(combinations)
			combinations.remove(combination)
			random_combinations.append(combination)

		for combination in random_combinations:	
			params = dict(zip(all_keys, combination))
			algorithm.set_params(**params)		
			accurasy = evaluate_algorithm(algorithm, features, labels, n_splits, standard_scale, min_max_scale)
			if (accurasy > best_accurasy):
				best_accurasy = accurasy
				best_algorithm = algorithm
		print(best_algorithm, len(random_combinations), best_accurasy, (time()-start)/60)
	return best_algorithm


def evaluate_algorithm(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	accurasies = 0
	kf = KFold(n_splits=n_splits, random_state=42)

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