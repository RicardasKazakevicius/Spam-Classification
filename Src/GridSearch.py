from Evaluation import get_algorithm_with_best_grid_params

def MLP_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):

	param_grids = []
	param_grids.append({
		'solver': ['adam'], 
		'hidden_layer_sizes': [(100,), (500,), (1000,), (100,100,100), (500,500,500), (1000,1000,1000)],
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		'learning_rate_init': [0.001, 1e-5],
		'max_iter': [200, 1000]
	})

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def LinearSVC_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	param_grids = []
	param_grids.append({
		'penalty': ['l2'],
		'loss': ['squared_hinge'], 
		'dual': [False],
		'C': [0.1, 0.5, 1, 1.5, 2., 2.5, 3., 5.],
		'multi_class': ['ovr', 'crammer_singer'],
		'max_iter': [1000, 2000]
	})
	param_grids.append({
		'penalty': ['l2'],
		'loss': ['hinge'],
		'dual': [True],
		'C': [0.1, 0.5, 1, 1.5, 2., 2.5, 3., 5.],
		'multi_class': ['ovr', 'crammer_singer'],
		'max_iter': [1000, 2000]
	})
	param_grids.append({
		'penalty': ['l1'],
		'loss': ['squared_hinge'],
		'dual': [False],
		'C': [0.1, 0.5, 1, 1.5, 2., 2.5, 3., 5.],
		'multi_class': ['ovr', 'crammer_singer'],
		'max_iter': [1000, 2000]
	})

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def KNeighbours_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):

	param_grids = []
	param_grids.append({
		'n_neighbors': [3, 7, 11, 15, 19, 25],
		'weights': ['uniform', 'distance'],
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
		'p': [1, 2]
	})

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def LogisticRegression_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	param_grids = []
	param_grids.append({ 
		'penalty': ['l2'],
		'dual': [False],
		'C': [0.1, 0.5, 1, 1.5, 2., 2.5, 3., 5.],
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'max_iter': [100,1000]
	})
	param_grids.append({
		'penalty': ['l2'],
		'dual': [True],
		'C': [0.1, 0.5, 1, 1.5, 2., 2.5, 3., 5.],
		'solver': ['liblinear'],
		'max_iter': [100]
	})
	param_grids.append({
		'penalty': ['l1'],
		'dual': [False],
		'C': [0.1, 0.5, 1, 1.5, 2., 2.5, 3., 5.],
		'solver': ['liblinear', 'saga'],
		'max_iter': [100]
	})

	return get_algorithm_with_best_grid_params(algorithm, param_grids, features, labels, n_splits, standard_scale, min_max_scale)


def MultinomialNB_grid_search(algorithm, features, labels, n_splits, standard_scale, min_max_scale):
	
	param_grids = []
	param_grids.append({
		'alpha': [x*0.1 for x in range(0, 49)],
		'fit_prior': [True, False]
	})

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