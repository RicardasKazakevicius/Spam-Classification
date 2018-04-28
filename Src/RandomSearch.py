from Evaluation import get_algorithm_with_best_random_params

def MLP_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	random_hidden_layer_sizes = [x*100 for x in range(1, 11)]
	random_learning_rate_init = [10**(-x) for x in range(1, 6)]
	random_tol = [10**(-x) for x in range(1, 8)]
	random_max_iter = [x*100 for x in range(1, 12, 3)]
	
	param_grids = []
	param_grids.append({
		'solver': ['adam'], 
		'hidden_layer_sizes': random_hidden_layer_sizes,
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		'learning_rate_init': random_learning_rate_init,
		'max_iter': random_max_iter,
		'tol': random_tol
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)


def SVC_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):
	
	random_C = [1-0.1*x for x in range(-10, 10)]
	random_degree = [x for x in range(1, 9)]

	param_grids = []
	param_grids.append({
		'C': random_C,
		'kernel':  ['linear', 'rbf', 'sigmoid'],
		'decision_function_shape': ['ovo', 'ovr'],
		'probability': [True, False]
	})
	param_grids.append({
		'C': random_C,
		'kernel':  ['poly'],
		'degree': random_degree,
		'decision_function_shape': ['ovo', 'ovr'],
		'probability': [True, False]
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)


def KNeighbours_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	random_n_neighbors = [x for x in range(3, 26)]

	param_grids = []
	param_grids.append({
		'n_neighbors': random_n_neighbors,
		'weights': ['uniform', 'distance'],
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
		'p': [1, 2]
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)


def LogisticRegression_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	random_C = [1-0.1*x for x in range(-10, 10)]
	random_max_iter = [x*100 for x in range(1, 11, 2)]
	random_tol = [10**(-x) for x in range(1, 8)]

	param_grids = []
	param_grids.append({
		'penalty': ['l2'],
		'dual': [False],
		'C': random_C,
		'tol': random_tol,
		'fit_intercept': [True, False],
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'max_iter': random_max_iter
	})
	param_grids.append({
		'penalty': ['l2'],
		'dual': [True],
		'C': random_C,
		'tol': random_tol,
		'fit_intercept': [True, False],
		'solver': ['liblinear'],
		'max_iter': [100]
	})
	param_grids.append({
		'penalty': ['l1'],
		'dual': [False],
		'C': random_C,
		'tol': random_tol,
		'fit_intercept': [True, False],
		'solver': ['liblinear', 'saga'],
		'max_iter': [100]
	})

	return get_algorithm_with_best_random_params(algorithm, param_grids, features, labels, n_splits, n_iter, standard_scale, min_max_scale)


def MultinomialNB_random_search(algorithm, features, labels, n_splits, n_iter, standard_scale, min_max_scale):
	
	random_alpha = [2-0.1*x for x in range(-28, 20)]

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