from Evaluation import get_algorithm_with_best_random_params

def MLP_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	# random_hidden_layer_sizes = [(x*100,) for x in range(1, 22, 2)] + \
															# [(x*100,x*100) for x in range(1, 22, 2)] + \
															# [(x*100,x*100,x*100) for x in range(1, 22, 2)]
	random_hidden_layer_sizes = [(x*10,) for x in range(1, 10, 1)] + \
															[(x*10,x*10) for x in range(1, 10, 1)] + \
															[(x*10,x*10,x*10) for x in range(1, 10, 1)]
	random_learning_rate_init = [10**(-x) for x in range(1, 6)]
	random_max_iter = [x*100 for x in range(1, 22, 3)]
	
	param_grids = []
	param_grids.append({
		'solver': ['adam'], 
		'hidden_layer_sizes': random_hidden_layer_sizes,
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		'learning_rate_init': random_learning_rate_init,
		'max_iter': random_max_iter
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)


def LinearSVC_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	random_C = [x*0.1 for x in range(1, 50)]
	random_max_iter = [x for x in range(800, 2500, 200)]

	param_grids = []
	param_grids.append({
		'C': random_C,
		'penalty': ['l1'],
		'loss': ['squared_hinge'],
		'multi_class': ['ovr', 'crammer_singer'],
		'dual': [False],
		'max_iter': random_max_iter
	})
	param_grids.append({
		'C': random_C,
		'penalty': ['l2'],
		'loss': ['hinge'],
		'multi_class': ['ovr', 'crammer_singer'],
		'dual': [True],
		'max_iter': random_max_iter
	})
	param_grids.append({
		'C': random_C,
		'penalty': ['l1'],
		'loss': ['squared_hinge'],
		'multi_class': ['ovr', 'crammer_singer'],
		'dual': [False],
		'max_iter': random_max_iter
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)


def KNeighbours_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	random_n_neighbors = [x for x in range(3, 30)]

	param_grids = []
	param_grids.append({
		'n_neighbors': random_n_neighbors,
		'weights': ['uniform', 'distance'],
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
		'p': [1, 2]
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)


def LogisticRegression_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	random_C = [x*0.1 for x in range(1, 50)]
	random_max_iter = [x for x in range(100, 1500, 200)]

	param_grids = []
	param_grids.append({
		'penalty': ['l2'],
		'dual': [False],
		'C': random_C,
		'fit_intercept': [True, False],
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'max_iter': random_max_iter
	})
	param_grids.append({
		'penalty': ['l2'],
		'dual': [True],
		'C': random_C,
		'fit_intercept': [True, False],
		'solver': ['liblinear'],
		'max_iter': [100]
	})
	param_grids.append({
		'penalty': ['l1'],
		'dual': [False],
		'C': random_C,
		'fit_intercept': [True, False],
		'solver': ['liblinear', 'saga'],
		'max_iter': [100]
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)


def MultinomialNB_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):
	
	random_alpha = [x*0.01 for x in range(0, 500, 5)]

	param_grids = []
	param_grids.append({
		'alpha': random_alpha,
		'fit_prior': [True, False]
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)


def RandomForest_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	random_n_estimators = [10*x for x in range(1, 101, 2)]
	random_min_samples_split = [x for x in range(2, 11)]
	random_min_samples_leaf = [x for x in range(1, 10)]

	param_grids = []
	param_grids.append({
		'n_estimators': random_n_estimators,
		'criterion': ['gini', 'entropy'],
		'max_features': ['auto', 'log2', None],
		'min_samples_split': random_min_samples_split,
		'min_samples_leaf': random_min_samples_leaf,
		'oob_score': [True, False]
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)