from Evaluation import get_algorithm_with_best_grid_params

def MLP_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):

	param_grids = []
	param_grids.append({
		'solver': ['adam'], 
		'hidden_layer_sizes': [(100,), (500), (1000,)],
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		'learning_rate_init': [0.001, 1e-5],
		'max_iter': [200, 1000],
		'tol': [1e-4, 1e-7]
	})

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def LinearSVC_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	param_grids = []
	param_grids.append({
		'penalty': ['l2'],
		'loss': ['squared_hinge'], 
		'C': [0.1, 0.7, 1.3, 2.],
		'multi_class': ['ovr', 'crammer_singer'],
		'dual': [False],
		'max_iter': [1000, 1500],
		'tol': [1e-4, 1e-7]
	})
	param_grids.append({
		'penalty': ['l2'],
		'loss': ['hinge'],
		'dual': [True],
		'C': [0.1, 0.7, 1.3, 2.],
		'multi_class': ['ovr', 'crammer_singer'],
		'max_iter': [1000, 1500],
		'tol': [1e-4, 1e-7]
	})
	param_grids.append({
		'penalty': ['l1'],
		'loss': ['squared_hinge'],
		'dual': [False],
		'C': [0.1, 0.7, 1.3, 2.],
		'multi_class': ['ovr', 'crammer_singer'],
		'max_iter': [1000, 1500],
		'tol': [1e-4, 1e-7]
	})

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def SVC_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):

	param_grids = []
	param_grids.append({
		'C': [0.1, 0.7, 1.3, 2.],
		'kernel':  ['linear', 'rbf', 'sigmoid'],
		'decision_function_shape': ['ovo', 'ovr'],
		'probability': [True, False]
	})
	param_grids.append({
		'C': [0.1, 0.7, 1.3, 2.],
		'kernel':  ['poly'],
		'degree': [3, 5, 7],
		'decision_function_shape': ['ovo', 'ovr'],
		'probability': [True, False]
	})

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def DecisionTree_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):

	param_grids = {
		'criterion': ['gini', 'entropy'],
		'splitter': ['best', 'random'],
		'max_features': ['auto', 'log2', None],
		'presort': [False, True],
		'min_samples_split': [2, 9],
		'min_samples_leaf': [1, 9]
	}

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def KNeighbours_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):

	param_grids = {
		'n_neighbors': [3, 7, 11, 15, 19, 25],
		'weights': ['uniform', 'distance'],
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
		'p': [1, 2]
	}

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def LogisticRegression_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	param_grids = []
	param_grids.append({ 
		'penalty': ['l2'],
		'tol': [1e-4, 1e-7],
		'C': [0.1, 0.7, 1.3, 2.],
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'max_iter': [100, 450]
	})
	param_grids.append({
		'penalty': ['l2'],
		'dual': [True],
		'tol': [1e-4, 1e-7],
		'C': [0.1, 0.7, 1.3, 2.],
		'solver': ['liblinear']
	})
	param_grids.append({
		'penalty': ['l1'],
		'dual': [False],
		'C': [0.1, 0.7, 1.3, 2.],
		'solver': ['liblinear', 'saga']
	})

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def MultinomialNB_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	param_grids = {
		'alpha': [2-0.1*x for x in range(-28, 20)],
		'fit_prior': [True, False]
	}

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def BernoulliNB_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	param_grids = {
		'alpha': [0.1, 0.3, 0.5, 1.0, 1.5, 2.0],
		'binarize': [None, 0.0, 0.1, 0.5, 0.8, 1.0, 1.5, 2.0],
		'fit_prior': [True, False]
	}

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def AdaBoost_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	param_grids = {
		'n_estimators': [50, 100, 250, 400, 550, 700, 850, 1000],
		'learning_rate': [0.01, 0.1, 0.5, 1., 1.5, 2.],
		'algorithm': ['SAMME', 'SAMME.R'],
	}

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)
	

def RandomForest_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):

	param_grids = []
	param_grids.append({
		'n_estimators': [10, 100],
		'criterion': ['gini', 'entropy'],
		'max_features': ['auto', 'log2', None],
		'min_samples_split': [2, 9],
		'min_samples_leaf': [1, 9],
		'oob_score': [True, False]
	})

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)