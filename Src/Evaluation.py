from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def evaluate(algorithms, features, labels, number_of_tests, standard_scale=False, min_max_scale=False):
	
	for t in range(number_of_tests):
		print('Test: %d' % (t+1))
		train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, shuffle=True)

		if (standard_scale):
			train_features, test_features = scale_standard(train_features, test_features)

		elif (min_max_scale):
			train_features, test_features = scale_min_max(train_features, test_features)			
		
		algorithms.fit(train_features, train_labels)		
		algorithms.predict(test_features, test_labels)
		algorithms.set_standings()

	return algorithms


# Scaling features with mean zero and standard deviation one
def scale_standard(train_features, test_features):
	scaler = StandardScaler()
	train_features = scaler.fit_transform(train_features)
	test_features = scaler.fit_transform(test_features)
	return train_features, test_features

# Scaling features with values between zero and one
def scale_min_max(train_features, test_features):
	scaler = MinMaxScaler(copy=False)
	train_features = scaler.fit_transform(train_features)
	test_features = scaler.fit_transform(test_features)
	return train_features, test_features