from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from GridSearch import *
from RandomSearch import *

class Algorithms:

	algorithms = []

	def append(self, algorithm):
		self.algorithms.append(algorithm)
		print(algorithm.get_name())

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
		for i in range(len(self.algorithms)):
			position = 1
			for j in range(len(self.algorithms)):
				if (self.algorithms[i].get_accurasies()[-1] < self.algorithms[j].get_accurasies()[-1]):
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


def get_random_search_tuned_algorithms(features, labels, n_splits, n_iter, standard_scale, min_max_scale):

	algorithms = Algorithms()
	# algorithms.append(Algorithm(MLP_random_search(MLPClassifier(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(LinearSVC_random_search(LinearSVC(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(SVC_random_search(SVC(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(DecisionTree_random_search(DecisionTreeClassifier(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(KNeighbours_random_search(KNeighborsClassifier(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(LogisticRegression_random_search(LogisticRegression(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(MultinomialNB_random_search(MultinomialNB(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(BernoulliNB_random_search(BernoulliNB(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(AdaBoost_random_search(AdaBoostClassifier(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(RandomForest_random_search(RandomForestClassifier(), features, labels, n_splits, n_iter, standard_scale, min_max_scale)))
	return algorithms


def get_grid_search_tuned_algorithms(features, labels, n_splits, standard_scale, min_max_scale):

	algorithms = Algorithms()
	# algorithms.append(Algorithm(MLP_grid_search(MLPClassifier(), features, labels, n_splits, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(LinearSVC_grid_search(LinearSVC(), features, labels, n_splits, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(SVC_grid_search(SVC(), features, labels, n_splits, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(DecisionTree_grid_search(DecisionTreeClassifier(), features, labels, n_splits, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(KNeighbours_grid_search(KNeighborsClassifier(), features, labels, n_splits, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(LogisticRegression_grid_search(LogisticRegression(), features, labels, n_splits, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(MultinomialNB_grid_search(MultinomialNB(), features, labels, n_splits, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(BernoulliNB_grid_search(BernoulliNB(), features, labels, n_splits, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(AdaBoost_grid_search(AdaBoostClassifier(), features, labels, n_splits, standard_scale, min_max_scale)))
	algorithms.append(Algorithm(RandomForest_grid_search(RandomForestClassifier(), features, labels, n_splits, standard_scale, min_max_scale)))
	return algorithms


