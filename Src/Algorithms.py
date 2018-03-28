from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from numpy.random import RandomState
import numpy as np

def get_algorithms():
	
	algorithms = []
	
	algorithms.append({'Name': 'MLP', 'Algo': MLPClassifier(), 'Accuracy': [], 'Standings': []})

	algorithms.append({'Name': 'LinearSVC', 'Algo': LinearSVC(), 'Accuracy': [], 'Standings': []})

	algorithms.append({'Name': 'SVC', 'Algo': SVC(), 'Accuracy': [], 'Standings': []})
	
	algorithms.append({'Name': 'DecisionTree', 'Algo': DecisionTreeClassifier(), 'Accuracy': [], 'Standings': []})
	
	algorithms.append({'Name': 'k-NN', 'Algo': KNeighborsClassifier(), 'Accuracy': [], 'Standings': []})
	
	algorithms.append({'Name': 'LogisticRegression', 'Algo': LogisticRegression(), 'Accuracy': [], 'Standings': []})

	algorithms.append({'Name': 'MultinomialNB', 'Algo': MultinomialNB(), 'Accuracy': [], 'Standings': []})
	algorithms.append({'Name': 'BernoulliNB', 'Algo': BernoulliNB(), 'Accuracy': [], 'Standings': []})

	algorithms.append({'Name': 'AdaBoost', 'Algo': AdaBoostClassifier(), 'Accuracy': [], 'Standings': []})

	algorithms.append({'Name': 'RandomForest', 'Algo': RandomForestClassifier(), 'Accuracy': [], 'Standings': []})	
	
	return algorithms


def get_tuned_algorithms(features, labels, n_jobs, verbose, cv):

	algorithms = []

	print('MPL')
	algo, params = tuned_MLP(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'MLP', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})
	
	print('LinearSVC')
	algo, params = tuned_LinearSVC(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'LinearSVC', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})

	print('SVC')
	algo, params = tuned_SVC(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'SVC', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})

	print('DecisionTree')
	algo, params = tuned_DecisionTree(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'DecisionTree', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})
	
	print('KNeighbors')
	algo, params = tuned_KNeighbours(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'k-NN', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})
	
	print('LogisticRegression')
	algo, params = tuned_LogisticRegression(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'LogisticRegression', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})

	print('MultinomialNB')
	algo, params = tuned_MultinomialNB(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'MultinomialNB', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})
	
	print('BernoulliNB')
	algo, params = tuned_BernoulliNB(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'BernoulliNB', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})
	
	print('AdaBoost')
	algo, params = tuned_AdaBoost(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'AdaBoost', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})

	print('RandomForest')
	algo, params = tuned_RandomForest(features, labels, n_jobs, verbose, cv)
	algorithms.append({'Name': 'RandomForest', 'Algo': algo, 'Accuracy': [], 'Standings': [], 'Params_info': params})
	
	return algorithms


def tuned_MLP(features, labels, n_jobs, verbose, cv):

	# solver=lbfgs
	param_grid_lbfgs = {
		'solver': ['lbfgs'], 
		'hidden_layer_sizes': [(100,), (200,), (300,), (500,)],
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		'max_iter': [100000],
		# 'tol': [1e-5, 1e-4],
		# 'warm_start': [True, False],
	}

	# solver=sgd
	param_grid_sgd_with_early_stopping = {
		'solver': ['sgd'],
		'hidden_layer_sizes': [(100,), (200,), (300,), (500,)],
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		# 'learning_rate': ['constant', 'invscaling', 'adaptive'],
		# 'learning_rate_init': [0.00001, 0.0001],
		'max_iter': [100000],
		# 'power_t': [0.1, 0.5, 1],
		# 'tol': [1e-5, 1e-4],
		# 'warm_start': [True, False],
		# 'momentum': [0.5, 0.9],
		# 'nesterovs_momentum': [True, False],
		# 'early_stopping': [True],
		# 'validation_fraction': [0.05, 0.1, 0.2],
	}

	# solver=sgd
	param_grid_sgd_without_early_stopping = {
		'solver': ['sgd'],
		'hidden_layer_sizes': [(100,), (200,), (300,), (500,)],
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		# 'learning_rate': ['constant', 'invscaling', 'adaptive'],
		# 'learning_rate_init': [0.00001, 0.0001],
		'max_iter': [100000],
		# 'power_t': [0.1, 0.5, 1],
		# 'tol': [1e-5, 1e-4],
		# 'warm_start': [True, False],
		# 'momentum': [0.5, 0.9],
		# 'nesterovs_momentum': [True, False],
		# 'early_stopping': [False],
	}

	# solver=adam
	param_grid_adam = {
		'solver': ['adam'], 
		'hidden_layer_sizes': [(100,), (200,), (300,), (500,)],
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		# 'learning_rate_init': [0.00001, 0.0001],
		'max_iter': [100000],
		# 'tol': [1e-5, 1e-4],
		# 'warm_start': [True, False],
		# 'early_stopping': [True, False],
		# 'validation_fraction': [0.05, 0.1, 0.2],
		# 'epsilon': [1e-9, 1e-8]
	}

	model_sgd_1 = GridSearchCV(MLPClassifier(), param_grid=param_grid_sgd_with_early_stopping, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_sgd_1.fit(features, labels)
	
	# model_sgd_2 = GridSearchCV(MLPClassifier(), param_grid=param_grid_sgd_without_early_stopping, n_jobs=n_jobs, verbose=verbose, cv=cv)
	# model_sgd_2.fit(features, labels)

	model_lbfgs = GridSearchCV(MLPClassifier(), param_grid=param_grid_lbfgs, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_lbfgs.fit(features, labels)	

	model_adam = GridSearchCV(MLPClassifier(), param_grid=param_grid_adam, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_adam.fit(features, labels)

	models = []
	models.append(model_sgd_1)
	# models.append(model_sgd_2)
	models.append(model_lbfgs)
	models.append(model_adam)

	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]

	return best_model.best_estimator_, get_params_info(best_model.param_grid)


def tuned_LinearSVC(features, labels, n_jobs, verbose, cv):

	param_grid_1 = {
		'penalty': ['l2'],
		'loss': ['squared_hinge'], 
		'C': [0.1, 0.5, 1, 1.5],
		'multi_class': ['ovr', 'crammer_singer'],
		'dual': [False],
		# 'max_iter': [10000],
		# 'tol': [1e-5, 1e-4]
	}

	param_grid_2 = {
		'penalty': ['l2'],
		'loss': ['hinge'],
		'dual': [True],
		'C': [0.1, 0.5, 1, 1.5],
		'multi_class': ['ovr', 'crammer_singer'],
		# 'max_iter': [10000]
		# 'tol': [1e-5, 1e-4]
	}

	param_grid_3 = {
		'penalty': ['l1'],
		'loss': ['squared_hinge'],
		'dual': [False],
		'C': [0.1, 0.5, 1, 1.5],
		'multi_class': ['ovr', 'crammer_singer'],
		# 'max_iter': [10000],
		# 'tol': [1e-5, 1e-4]
	}

	model_1 = GridSearchCV(LinearSVC(), param_grid=param_grid_1, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_1.fit(features, labels)

	model_2 = GridSearchCV(LinearSVC(), param_grid=param_grid_2, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_2.fit(features, labels)

	model_3 = GridSearchCV(LinearSVC(), param_grid=param_grid_3, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_3.fit(features, labels)

	models = []
	models.append(model_1)
	models.append(model_2)
	models.append(model_3)

	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
	return best_model.best_estimator_, get_params_info(best_model.param_grid)
	

def tuned_SVC(features, labels, n_jobs, verbose, cv):

	param_grid_1 = {
		'C': [0.1, 0.5, 1, 1.5],
		'kernel':  ['linear', 'rbf', 'sigmoid'],
		'decision_function_shape': ['ovo', 'ovr'],
		# 'shrinking': [True, False],
		# 'probability': [True, False],		
		# 'tol': [1e-4, 1e-3]
	},

	param_grid_2 = {
		'C': [0.1, 0.5, 1, 1.5],
		'kernel':  ['poly'],
		'degree': [1, 3, 5],
		'decision_function_shape': ['ovo', 'ovr'],
		# 'shrinking': [True, False],
		# 'probability': [true, false],
		# 'tol': [1e-4, 1e-3]
	}

	model_1 = GridSearchCV(SVC(), param_grid=param_grid_1, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_1.fit(features, labels)
	
	model_2 = GridSearchCV(SVC(), param_grid=param_grid_2, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_2.fit(features, labels)
	
	models = []
	models.append(model_1)
	models.append(model_2)
	
	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
	return best_model.best_estimator_, get_params_info(best_model.param_grid[0])


def tuned_KNeighbours(features, labels, n_jobs, verbose, cv):

	param_grid = {
		'n_neighbors': [3, 5, 7, 11, 15, 25],
		'weights': ['uniform', 'distance'],
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
		'p': [1, 2]
	}

	model = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model.fit(features, labels)

	return model.best_estimator_, get_params_info(param_grid)


def tuned_LogisticRegression(features, labels, n_jobs, verbose, cv):

	param_grid_1 = {
		'penalty': ['l2'],
		# 'tol': [1e-5, 1e-4],
		'C': [0.1, 0.5, 1, 1.5],
		'fit_intercept': [True, False],
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'max_iter': [100000]
	}

	param_grid_2 = {
		'penalty': ['l2'],
		'dual': [True],
		# 'tol': [1e-5, 1e-4],
		'C': [0.1, 0.5, 1, 1.5],
		'fit_intercept': [True, False],
		'solver': ['liblinear']
	}

	param_grid_3 = {
		'penalty': ['l1'],
		'dual': [False],
		# 'tol': [1e-5, 1e-4],
		'C': [0.1, 0.5, 1, 1.5],
		'fit_intercept': [True, False],
		'solver': ['liblinear', 'saga']
	}

	model_1 = GridSearchCV(LogisticRegression(), param_grid=param_grid_1, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_1.fit(features, labels)

	model_2 = GridSearchCV(LogisticRegression(), param_grid=param_grid_2, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_2.fit(features, labels)

	model_3 = GridSearchCV(LogisticRegression(), param_grid=param_grid_3, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_3.fit(features, labels)

	models = []
	models.append(model_1)
	models.append(model_2)
	models.append(model_3)
	
	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
	return best_model.best_estimator_, get_params_info(best_model.param_grid)


def tuned_DecisionTree(features, labels, n_jobs, verbose, cv):

	param_grid = {
		'criterion': ['gini', 'entropy'],
		'splitter': ['best', 'random'],
		'max_features': ['auto', 'log2', None],
		'presort': [False, True],
		'min_samples_split': [2, 4, 6]
	}

	model = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model.fit(features, labels)

	return model.best_estimator_, get_params_info(param_grid)


def tuned_AdaBoost(features, labels, n_jobs, verbose, cv):

	param_grid = {
		'n_estimators': [50, 100, 250, 500, 1000],
		'learning_rate': [0.01, 0.1, 0.5, 1., 1.5],
		'algorithm': ['SAMME', 'SAMME.R'],
	}

	model = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model.fit(features, labels)

	return model.best_estimator_, get_params_info(param_grid)


def tuned_RandomForest(features, labels, n_jobs, verbose, cv):

	param_grid_1 = {
		'n_estimators': [50, 100, 250, 500],
		'criterion': ['gini', 'entropy'],
		'max_features': ['auto', 'log2', None],
		'min_samples_split': [2, 4, 6],
		'oob_score': [True, False]
	}

	param_grid_2 = {
		'n_estimators': [50, 100, 250, 500],
		'criterion': ['gini', 'entropy'],
		'max_features': ['auto', 'log2', None],
		'min_samples_split': [2, 4, 6],
		'bootstrap': [False],
		'oob_score': [False]
	}

	model_1 = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_1, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_1.fit(features, labels)

	model_2 = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_2, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model_2.fit(features, labels)

	models = []
	models.append(model_1)
	models.append(model_2)
	
	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
	return best_model.best_estimator_, get_params_info(best_model.param_grid)


def tuned_BernoulliNB(features, labels, n_jobs, verbose, cv):

	param_grid = {
		'alpha': [1e-10, 0.001, 0.01, 0.1, 1.0, 1.5],
		'binarize': [None, 0.0, 0.0001, 0.01, 0.1, 1.0],
		'fit_prior': [True, False]
	}

	model = GridSearchCV(BernoulliNB(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model.fit(features, labels)

	return model.best_estimator_, get_params_info(param_grid)


def tuned_MultinomialNB(features, labels, n_jobs, verbose, cv):
	
	param_grid = {
		'alpha': [1e-10, 0.001, 0.01, 0.1, 1.0, 1.5],
		'fit_prior': [True, False]
	}

	model = GridSearchCV(MultinomialNB(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
	model.fit(features, labels)

	return model.best_estimator_, get_params_info(param_grid)


def get_params_info(param_grid):

	i = 0
	number_of_params = np.zeros(len(param_grid))

	for key in param_grid.values():
		number_of_params[i] = len(key)
		i += 1

	return number_of_params
