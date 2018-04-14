import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def extract_features_and_labels(data_directory, dictionary, mails_count): 
    
	docID = 0
	features = np.zeros((mails_count, len(dictionary)))
	labels = np.zeros(mails_count)

	directories = [os.path.join(data_directory, f) for f in os.listdir(data_directory)] 
	for directory in directories:
	    innerDirectories = [os.path.join(directory, f) for f in os.listdir(directory)]
	    for innerDirectory in innerDirectories:
	        emails = [os.path.join(innerDirectory, f) for f in os.listdir(innerDirectory)]
	        for mail in emails:
	            with open(mail) as m:
	                all_words = []
	                for line in m:
	                    words = line.split()
	                    all_words += words
	                
	                for word in all_words:
	                  wordID = 0
	                  for i,d in enumerate(dictionary):
	                    if d[0] == word:
	                      wordID = i
	                      features[docID, wordID] = all_words.count(word)
	            
	            # If file or any folder name contains word 'spam' or 'spmsg' then set label as true with value 1
	            labels[docID] = int(mail.split('.')[-2] == 'spam') or 'spmsg' in mail or 'spam' in mail
	            docID = docID + 1   

	return features, labels