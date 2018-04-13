from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from numpy.random import RandomState
import numpy as np
import scipy as sp
import random

class Algorithms:

	algorithms = []

	def append(self, algorithm):
		self.algorithms.append(algorithm)

	# Training all algorithms
	def fit(self, train_features, train_labels):
		for algo in self.algorithms:
			algo.get().fit(train_features, train_labels)

	# Predicting and saving accurasies
	def predict(self, test_features, test_labels):
		for algo in self.algorithms:
			prediction = algo.get().predict(test_features)
			algo.accurasies.append(accuracy_score(test_labels, prediction)*100)

	# Counting how many times each algorithm took certain position comparing with rest algorithms accurasies
	def set_standings(self):		
		for t in range(len(self.algorithms[0].accurasies)):
			for i in range(len(self.algorithms)):
				position = 1
				for j in range(len(self.algorithms)):
					if (self.algorithms[i].get_accurasies()[t] < self.algorithms[j].get_accurasies()[t]):
						position += 1

				self.algorithms[i].standings.append(position)

	def get(self):
		return self.algorithms


class Algorithm:

	def __init__(self, algorithm):
		self.algorithm = algorithm
		self.name = str(algorithm).split('(')[0]
		self.accurasies = []
		self.standings = []

	def get(self):
		return self.algorithm

	def get_name(self):
		return self.name

	def get_accurasies(self):
		return self.accurasies

	def get_standings(self):
		return self.standings


def get_algorithms():
	
	algorithms = Algorithms()
	algorithms.append(Algorithm(MLPClassifier()))
	algorithms.append(Algorithm(LinearSVC()))
	algorithms.append(Algorithm(SVC()))
	algorithms.append(Algorithm(DecisionTreeClassifier()))
	algorithms.append(Algorithm(KNeighborsClassifier()))
	algorithms.append(Algorithm(LogisticRegression()))
	algorithms.append(Algorithm(MultinomialNB()))
	algorithms.append(Algorithm(BernoulliNB()))
	algorithms.append(Algorithm(AdaBoostClassifier()))
	algorithms.append(Algorithm(RandomForestClassifier()))
	return algorithms


def get_tuned_algorithms(features, labels, n_jobs, verbose, n_iter, cv):

	algorithms = Algorithms()
	algorithms.append(tuned_MLP(features, labels, n_jobs, cv, n_iter, verbose))
	algorithms.append(Algorithm(tuned_LinearSVC(features, labels, n_jobs, cv, n_iter, verbose)))
	algorithms.append(Algorithm(tuned_SVC(features, labels, n_jobs, cv, n_iter, verbose)))
	algorithms.append(Algorithm(tuned_DecisionTree(features, labels, n_jobs, cv, n_iter, verbose)))
	algorithms.append(Algorithm(tuned_KNeighbours(features, labels, n_jobs, cv, n_iter, verbose)))
	algorithms.append(Algorithm(tuned_LogisticRegression(features, labels, n_jobs, cv, n_iter, verbose)))
	algorithms.append(Algorithm(tuned_MultinomialNB(features, labels, n_jobs, cv, n_iter, verbose)))
	algorithms.append(Algorithm(tuned_BernoulliNB(features, labels, n_jobs, cv, n_iter, verbose)))
	algorithms.append(Algorithm(tuned_AdaBoost(features, labels, n_jobs, cv, n_iter, verbose)))
	algorithms.append(Algorithm(tuned_RandomForest(features, labels, n_jobs, cv, n_iter, verbose)))
	return algorithms


def tuned_MLP(features, labels, n_jobs, cv, n_iter, verbose):

	random_hidden_layer_sizes = [x*100 for x in range(1, 11)]
	random_learning_rate_init = [10**(-x) for x in range(1, 6)]
	random_tol = [10**(-x) for x in range(1, 8)]

	# solver=lbfgs
	param_dist = []
	param_dist.append({
		'solver': ['lbfgs'], 
		'hidden_layer_sizes': random_hidden_layer_sizes,
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		'max_iter': [100000],
		'tol': random_tol,
		# 'warm_start': [True, False],
	})

	# solver=sgd
	param_dist.append({
		'solver': ['sgd'],
		'hidden_layer_sizes': random_hidden_layer_sizes,
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		# 'learning_rate': ['constant', 'invscaling', 'adaptive'],
		'learning_rate_init': random_learning_rate_init,
		'max_iter': [100000],
		# 'power_t': [0.1, 0.5, 1],
		'tol': random_tol,
		# 'warm_start': [True, False],
		# 'momentum': [0.5, 0.9],
		# 'nesterovs_momentum': [True, False],
		# 'early_stopping': [True],
		# 'validation_fraction': [0.05, 0.1, 0.2],
	})

	# solver=sgd
	param_dist.append({
		'solver': ['sgd'],
		'hidden_layer_sizes': random_hidden_layer_sizes,
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		# 'learning_rate': ['constant', 'invscaling', 'adaptive'],
		'learning_rate_init': random_learning_rate_init,
		'max_iter': [100000],
		# 'power_t': [0.1, 0.5, 1],
		'tol': random_tol,
		# 'warm_start': [True, False],
		# 'momentum': [0.5, 0.9],
		# 'nesterovs_momentum': [True, False],
		# 'early_stopping': [False],
	})

	# solver=adam
	param_dist.append({
		'solver': ['adam'], 
		'hidden_layer_sizes': random_hidden_layer_sizes,
		'activation': ['identity', 'logistic', 'tanh', 'relu'],
		'learning_rate_init': random_learning_rate_init,
		'max_iter': [100000],
		'tol': random_tol,
		# 'warm_start': [True, False],
		# 'early_stopping': [True, False],
		# 'validation_fraction': [0.05, 0.1, 0.2],
		# 'epsilon': [1e-9, 1e-8]
	})

	models = []
	for i in range(len(param_dist)):
		model = RandomizedSearchCV(MLPClassifier(), param_dist[i], n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
		model.fit(features, labels)
		models.append(model)

	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]

	return best_model.best_estimator_


def tuned_KNeighbours(features, labels, n_jobs, cv, n_iter, verbose):

	random_n_neighbors = sp.stats.randint(3, 25)

	param_distributions = {
		'n_neighbors': random_n_neighbors,
		'weights': ['uniform', 'distance'],
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
		'p': [1, 2]
	}

	model = RandomizedSearchCV(KNeighborsClassifier(), param_distributions, n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
	model.fit(features, labels)

	return model.best_estimator_


def tuned_LinearSVC(features, labels, n_jobs, cv, n_iter, verbose):

	random_max_iter = [x*100 for x in range(5, 15)]
	random_C = [1-0.1*x for x in range(-10, 10)]
	random_tol = [10**(-x) for x in range(1, 8)]
	
	param_dist = []
	param_dist.append({
		'C': random_C,
		'tol': random_tol,
		'penalty': ['l1'],
		'loss': ['squared_hinge'],
		'multi_class': ['ovr', 'crammer_singer'],
		'dual': [False],
		'max_iter': random_max_iter
	})

	param_dist.append({
		'C': random_C,
		'tol': random_tol,
		'penalty': ['l2'],
		'loss': ['hinge'],
		'multi_class': ['ovr', 'crammer_singer'],
		'dual': [True],
		'max_iter': random_max_iter
	})

	param_dist.append({
		'C': random_C,
		'tol': random_tol,
		'penalty': ['l1'],
		'loss': ['squared_hinge'],
		'multi_class': ['ovr', 'crammer_singer'],
		'dual': [False],
		'max_iter': random_max_iter
	})

	models = []
	for i in range(len(param_dist)):
		model = RandomizedSearchCV(LinearSVC(), param_dist[i], n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
		model.fit(features, labels)
		models.append(model)

	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]

	return best_model.best_estimator_

# Su antru dist ilgai dirba kazkodel
def tuned_SVC(features, labels, n_jobs, cv, n_iter, verbose):
	
	random_C = [1-0.1*x for x in range(-10, 10)]
	random_degree = sp.stats.randint(1, 8)
	random_tol = [10**(-x) for x in range(1, 8)]

	param_dist = []
	param_dist.append({
		'C': random_C,
		'kernel':  ['linear', 'rbf', 'sigmoid'],
		'decision_function_shape': ['ovo', 'ovr'],
		'shrinking': [True, False],
		'probability': [True, False],		
		'tol': random_tol
	})

	param_dist.append({
		'C': random_C,
		'kernel':  ['poly'],
		'degree': random_degree,
		'decision_function_shape': ['ovo', 'ovr'],
		'shrinking': [True, False],
		'probability': [True, False],
		'tol': random_tol
	})

	models = []
	for i in range(len(param_dist)):
		model = RandomizedSearchCV(SVC(), param_dist[i], n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
		model.fit(features, labels)
		models.append(model)
	
	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
	return best_model.best_estimator_


def tuned_LogisticRegression(features, labels, n_jobs, cv, n_iter, verbose):

	random_C = [1-0.1*x for x in range(-10, 10)]
	random_max_iter = [x*50 for x in range(1, 10)] #100
	random_tol = [10**(-x) for x in range(1, 8)]

	param_dist = []
	param_dist.append({
		'penalty': ['l2'],
		'C': random_C,
		'tol': random_tol,
		'fit_intercept': [True, False],
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'max_iter': random_max_iter
	})

	param_dist.append({
		'penalty': ['l2'],
		'dual': [True],
		'C': random_C,
		'tol': random_tol,
		'fit_intercept': [True, False],
		'solver': ['liblinear']
	})

	param_dist.append({
		'penalty': ['l1'],
		'dual': [False],
		'C': random_C,
		'tol': random_tol,
		'fit_intercept': [True, False],
		'solver': ['liblinear', 'saga']
	})

	models = []
	for i in range(len(param_dist)):
		model = RandomizedSearchCV(LogisticRegression(), param_dist[i], n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
		model.fit(features, labels)
		models.append(model)

	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]

	return best_model.best_estimator_


def tuned_DecisionTree(features, labels, n_jobs, cv, n_iter, verbose):

	param_dist = {
		'criterion': ['gini', 'entropy'],
		'splitter': ['best', 'random'],
		'max_features': ['auto', 'log2', None],
		'presort': [False, True],
		'min_samples_split': [2, 4, 6]
	}

	model = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
	model.fit(features, labels)

	return model.best_estimator_


def tuned_AdaBoost(features, labels, n_jobs, cv, n_iter, verbose):

	random_n_estimators = [10*x for x in range(5, 10)]
	random_learning_rate = [1-x*0.1 for x in range(-10, 10)]

	param_dist = {
		'n_estimators': random_n_estimators,
		'learning_rate': random_learning_rate,
		'algorithm': ['SAMME', 'SAMME.R'],
	}

	model = RandomizedSearchCV(AdaBoostClassifier(), param_dist, n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
	model.fit(features, labels)

	return model.best_estimator_


def tuned_RandomForest(features, labels, n_jobs, cv, n_iter, verbose):

	random_n_estimators = [10*x for x in range(1, 100)]

	param_dist = []
	param_dist.append({
		'n_estimators': random_n_estimators,
		'criterion': ['gini', 'entropy'],
		'max_features': ['auto', 'log2', None],
		'min_samples_split': [2, 4, 6],
		'oob_score': [True, False]
	})

	param_dist.append({
		'n_estimators': random_n_estimators,
		'criterion': ['gini', 'entropy'],
		'max_features': ['auto', 'log2', None],
		'min_samples_split': [2, 4, 6],
		'bootstrap': [False],
		'oob_score': [False]
	})

	models = []
	for i in range(len(param_dist)):
		model = RandomizedSearchCV(RandomForestClassifier(), param_dist[i], n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
		model.fit(features, labels)
		models.append(model)

	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
	return best_model.best_estimator_


def tuned_BernoulliNB(features, labels, n_jobs, cv, n_iter, verbose):

	random_alpha = [10**(-x) for x in range(8)]
	random_binarize = []

	param_dist = {
		'alpha': random_alpha,
		'binarize': [None, 0.0, 0.0001, 0.01, 0.1, 1.0],
		'fit_prior': [True, False]
	}

	model = RandomizedSearchCV(BernoulliNB(), param_dist, n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
	model.fit(features, labels)

	return model.best_estimator_


def tuned_MultinomialNB(features, labels, n_jobs, cv, n_iter, verbose):
	
	random_alpha = [10**(-x) for x in range(8)]

	param_dist = {
		'alpha': random_alpha,
		'fit_prior': [True, False]
	}

	model = RandomizedSearchCV(MultinomialNB(), param_dist, n_iter=n_iter, n_jobs=n_jobs, cv=cv, verbose=verbose, return_train_score=False)
	model.fit(features, labels)

	return model.best_estimator_


# GridSearchCV
# def tuned_MLP(features, labels, n_jobs, cv, n_iter verbose):

# 	# solver=lbfgs
# 	param_grid_lbfgs = {
# 		'solver': ['lbfgs'], 
# 		'hidden_layer_sizes': [(100,), (200,), (300,), (500,)],
# 		'activation': ['identity', 'logistic', 'tanh', 'relu'],
# 		'max_iter': [100000],
# 		# 'tol': [1e-5, 1e-4],
# 		# 'warm_start': [True, False],
# 	}

# 	# solver=sgd
# 	param_grid_sgd_with_early_stopping = {
# 		'solver': ['sgd'],
# 		'hidden_layer_sizes': [(100,), (200,), (300,), (500,)],
# 		'activation': ['identity', 'logistic', 'tanh', 'relu'],
# 		# 'learning_rate': ['constant', 'invscaling', 'adaptive'],
# 		# 'learning_rate_init': [0.00001, 0.0001],
# 		'max_iter': [100000],
# 		# 'power_t': [0.1, 0.5, 1],
# 		# 'tol': [1e-5, 1e-4],
# 		# 'warm_start': [True, False],
# 		# 'momentum': [0.5, 0.9],
# 		# 'nesterovs_momentum': [True, False],
# 		# 'early_stopping': [True],
# 		# 'validation_fraction': [0.05, 0.1, 0.2],
# 	}

# 	# solver=sgd
# 	param_grid_sgd_without_early_stopping = {
# 		'solver': ['sgd'],
# 		'hidden_layer_sizes': [(100,), (200,), (300,), (500,)],
# 		'activation': ['identity', 'logistic', 'tanh', 'relu'],
# 		# 'learning_rate': ['constant', 'invscaling', 'adaptive'],
# 		# 'learning_rate_init': [0.00001, 0.0001],
# 		'max_iter': [100000],
# 		# 'power_t': [0.1, 0.5, 1],
# 		# 'tol': [1e-5, 1e-4],
# 		# 'warm_start': [True, False],
# 		# 'momentum': [0.5, 0.9],
# 		# 'nesterovs_momentum': [True, False],
# 		# 'early_stopping': [False],
# 	}

# 	# solver=adam
# 	param_grid_adam = {
# 		'solver': ['adam'], 
# 		'hidden_layer_sizes': [(100,), (200,), (300,), (500,)],
# 		'activation': ['identity', 'logistic', 'tanh', 'relu'],
# 		# 'learning_rate_init': [0.00001, 0.0001],
# 		'max_iter': [100000],
# 		# 'tol': [1e-5, 1e-4],
# 		# 'warm_start': [True, False],
# 		# 'early_stopping': [True, False],
# 		# 'validation_fraction': [0.05, 0.1, 0.2],
# 		# 'epsilon': [1e-9, 1e-8]
# 	}

# 	model_sgd_1 = GridSearchCV(MLPClassifier(), param_grid=param_grid_sgd_with_early_stopping, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_sgd_1.fit(features, labels)
	
# 	# model_sgd_2 = GridSearchCV(MLPClassifier(), param_grid=param_grid_sgd_without_early_stopping, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	# model_sgd_2.fit(features, labels)

# 	model_lbfgs = GridSearchCV(MLPClassifier(), param_grid=param_grid_lbfgs, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_lbfgs.fit(features, labels)	

# 	model_adam = GridSearchCV(MLPClassifier(), param_grid=param_grid_adam, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_adam.fit(features, labels)

# 	models = []
# 	models.append(model_sgd_1)
# 	# models.append(model_sgd_2)
# 	models.append(model_lbfgs)
# 	models.append(model_adam)

# 	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]

# 	return best_model.best_estimator_, get_params_info(best_model.param_grid)


# def tuned_LinearSVC(features, labels, n_jobs, verbose, cv):
# 	# Galima prideti parametru nes labai greit viska suskaiciuoja
# 	param_grid_1 = {
# 		'penalty': ['l2'],
# 		'loss': ['squared_hinge'], 
# 		'C': [0.1, 0.5, 1, 1.5],
# 		'multi_class': ['ovr', 'crammer_singer'],
# 		'dual': [False],
# 		# 'max_iter': [10000],
# 		# 'tol': [1e-5, 1e-4]
# 	}

# 	param_grid_2 = {
# 		'penalty': ['l2'],
# 		'loss': ['hinge'],
# 		'dual': [True],
# 		'C': [0.1, 0.5, 1, 1.5],
# 		'multi_class': ['ovr', 'crammer_singer'],
# 		# 'max_iter': [10000]
# 		# 'tol': [1e-5, 1e-4]
# 	}

# 	param_grid_3 = {
# 		'penalty': ['l1'],
# 		'loss': ['squared_hinge'],
# 		'dual': [False],
# 		'C': [0.1, 0.5, 1, 1.5],
# 		'multi_class': ['ovr', 'crammer_singer'],
# 		# 'max_iter': [10000],
# 		# 'tol': [1e-5, 1e-4]
# 	}

# 	model_1 = GridSearchCV(LinearSVC(), param_grid=param_grid_1, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_1.fit(features, labels)

# 	model_2 = GridSearchCV(LinearSVC(), param_grid=param_grid_2, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_2.fit(features, labels)

# 	model_3 = GridSearchCV(LinearSVC(), param_grid=param_grid_3, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_3.fit(features, labels)

# 	models = []
# 	models.append(model_1)
# 	models.append(model_2)
# 	models.append(model_3)

# 	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
# 	return best_model.best_estimator_, get_params_info(best_model.param_grid)
	

# def tuned_SVC(features, labels, n_jobs, verbose, cv):

# 	param_grid_1 = {
# 		'C': [0.1, 0.5, 1, 1.5],
# 		'kernel':  ['linear', 'rbf', 'sigmoid'],
# 		'decision_function_shape': ['ovo', 'ovr'],
# 		# 'shrinking': [True, False],
# 		# 'probability': [True, False],		
# 		# 'tol': [1e-4, 1e-3]
# 	},

# 	param_grid_2 = {
# 		'C': [0.1, 0.5, 1, 1.5],
# 		'kernel':  ['poly'],
# 		'degree': [1, 3, 5],
# 		'decision_function_shape': ['ovo', 'ovr'],
# 		# 'shrinking': [True, False],
# 		# 'probability': [true, false],
# 		# 'tol': [1e-4, 1e-3]
# 	}

# 	model_1 = GridSearchCV(SVC(), param_grid=param_grid_1, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_1.fit(features, labels)
	
# 	model_2 = GridSearchCV(SVC(), param_grid=param_grid_2, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_2.fit(features, labels)
	
# 	models = []
# 	models.append(model_1)
# 	models.append(model_2)
	
# 	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
# 	return best_model.best_estimator_, get_params_info(best_model.param_grid[0])


# def tuned_KNeighbours(features, labels, n_jobs, verbose, cv):

# 	param_grid = {
# 		'n_neighbors': [3, 5, 7, 11, 15, 25],
# 		'weights': ['uniform', 'distance'],
# 		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
# 		'p': [1, 2]
# 	}

# 	model = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model.fit(features, labels)

# 	return model.best_estimator_, get_params_info(param_grid)


# def tuned_LogisticRegression(features, labels, n_jobs, verbose, cv):

# 	param_grid_1 = {
# 		'penalty': ['l2'],
# 		# 'tol': [1e-5, 1e-4],
# 		'C': [0.1, 0.5, 1, 1.5],
# 		'fit_intercept': [True, False],
# 		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
# 		'max_iter': [100000]
# 	}

# 	param_grid_2 = {
# 		'penalty': ['l2'],
# 		'dual': [True],
# 		# 'tol': [1e-5, 1e-4],
# 		'C': [0.1, 0.5, 1, 1.5],
# 		'fit_intercept': [True, False],
# 		'solver': ['liblinear']
# 	}

# 	param_grid_3 = {
# 		'penalty': ['l1'],
# 		'dual': [False],
# 		# 'tol': [1e-5, 1e-4],
# 		'C': [0.1, 0.5, 1, 1.5],
# 		'fit_intercept': [True, False],
# 		'solver': ['liblinear', 'saga']
# 	}

# 	model_1 = GridSearchCV(LogisticRegression(), param_grid=param_grid_1, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_1.fit(features, labels)

# 	model_2 = GridSearchCV(LogisticRegression(), param_grid=param_grid_2, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_2.fit(features, labels)

# 	model_3 = GridSearchCV(LogisticRegression(), param_grid=param_grid_3, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_3.fit(features, labels)

# 	models = []
# 	models.append(model_1)
# 	models.append(model_2)
# 	models.append(model_3)
	
# 	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
# 	return best_model.best_estimator_, get_params_info(best_model.param_grid)


# def tuned_DecisionTree(features, labels, n_jobs, verbose, cv):

# 	param_grid = {
# 		'criterion': ['gini', 'entropy'],
# 		'splitter': ['best', 'random'],
# 		'max_features': ['auto', 'log2', None],
# 		'presort': [False, True],
# 		'min_samples_split': [2, 4, 6]
# 	}

# 	model = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model.fit(features, labels)

# 	return model.best_estimator_, get_params_info(param_grid)


# def tuned_AdaBoost(features, labels, n_jobs, verbose, cv):

# 	param_grid = {
# 		'n_estimators': [50, 100, 250, 500, 1000],
# 		'learning_rate': [0.01, 0.1, 0.5, 1., 1.5],
# 		'algorithm': ['SAMME', 'SAMME.R'],
# 	}

# 	model = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model.fit(features, labels)

# 	return model.best_estimator_, get_params_info(param_grid)


# def tuned_RandomForest(features, labels, n_jobs, verbose, cv):

# 	param_grid_1 = {
# 		'n_estimators': [50, 100, 250, 500],
# 		'criterion': ['gini', 'entropy'],
# 		'max_features': ['auto', 'log2', None],
# 		'min_samples_split': [2, 4, 6],
# 		'oob_score': [True, False]
# 	}

# 	param_grid_2 = {
# 		'n_estimators': [50, 100, 250, 500],
# 		'criterion': ['gini', 'entropy'],
# 		'max_features': ['auto', 'log2', None],
# 		'min_samples_split': [2, 4, 6],
# 		'bootstrap': [False],
# 		'oob_score': [False]
# 	}

# 	model_1 = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_1, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_1.fit(features, labels)

# 	model_2 = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_2, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model_2.fit(features, labels)

# 	models = []
# 	models.append(model_1)
# 	models.append(model_2)
	
# 	best_model = sorted(models, key=lambda x: x.best_score_, reverse=True)[0]
	
# 	return best_model.best_estimator_, get_params_info(best_model.param_grid)


# def tuned_BernoulliNB(features, labels, n_jobs, verbose, cv):

# 	param_grid = {
# 		'alpha': [1e-10, 0.001, 0.01, 0.1, 1.0, 1.5],
# 		'binarize': [None, 0.0, 0.0001, 0.01, 0.1, 1.0],
# 		'fit_prior': [True, False]
# 	}

# 	model = GridSearchCV(BernoulliNB(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model.fit(features, labels)

# 	return model.best_estimator_, get_params_info(param_grid)


# def tuned_MultinomialNB(features, labels, n_jobs, verbose, cv):
	
# 	param_grid = {
# 		'alpha': [1e-10, 0.001, 0.01, 0.1, 1.0, 1.5],
# 		'fit_prior': [True, False]
# 	}

# 	model = GridSearchCV(MultinomialNB(), param_grid=param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv)
# 	model.fit(features, labels)

# 	return model.best_estimator_, get_params_info(param_grid)


def get_params_info(param_grid):

	i = 0
	number_of_params = np.zeros(len(param_grid))

	for key in param_grid.values():
		number_of_params[i] = len(key)
		i += 1

	return number_of_params
