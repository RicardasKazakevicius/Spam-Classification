from collections import Counter
from stemming.porter2 import stem
import os

def get_stop_words():
	
	with open('stopWords.txt', 'r') as file:  
		stop_words = file.readlines()
		for i in range(len(stop_words)):
			stop_words[i] = stop_words[i].replace('\n', '')

	return stop_words


def make_dictonary(data_directory, most_common_words, rm_stop_words, stemming, tokenization):

	all_words = []
	mails_count = 0

	directories = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]
	for directory in directories:
		inner_directories = [os.path.join(directory, f) for f in os.listdir(directory)]
		for inner_directory in inner_directories:
			emails = [os.path.join(inner_directory, f) for f in os.listdir(inner_directory)]
			for mail in emails:
				with open(mail) as m:
					mails_count += 1
					for line in m:
						words = line.split()
						all_words += words

	all_words = [w.lower() for w in all_words]
	
	# Making semantically similar words same
	if (stemming):
		for i in range(len(all_words)):
			all_words[i] = stem(all_words[i])

	dictionary = Counter(all_words)
	
	# Removing non string type and with length one items
	if (tokenization):
		for item in dictionary.keys():
			if item.isalpha() == False or len(item) == 1: 
				del dictionary[item]

	# Removing stop words
	if (rm_stop_words):
		stop_words = get_stop_words()
		for item in dictionary.keys():
			if item in stop_words: 
				del dictionary[item]
	
	if (most_common_words):
		dictionary = dictionary.most_common(most_common_words)
	
	return dictionary, mails_count