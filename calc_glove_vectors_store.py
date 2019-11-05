#  Required imports
import pandas as pd
import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn import svm
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score



fields = ['Event 1', 'Event 2', 'Label']
stop = set(('at', 'the', 'a', 'an', 'on', '(', ')', 'in', 'the', 'to', 'with', 'of', 'be', 'by', 'from'))


def embedAndConcatenate(df):


	# Load spacy model
	nlp = spacy.load('en_core_web_lg')

	data = np.zeros((df.shape[0], 2*len(nlp.vocab['dog'].vector)))
	# print(trainData.shape)

	for i in range(0, df.shape[0]):
		
		# Event1
		x = np.zeros(len(nlp.vocab['dog'].vector,))
		for j in range(0, len(df['Event 1'][i])):
			x = x + nlp.vocab[df['Event 1'][i][j]].vector
		x = x/len(df['Event 1'][i])

		# Event2
		y = np.zeros(len(nlp.vocab['dog'].vector,))
		for j in range(0, len(df['Event 2'][i])):
			y = y + nlp.vocab[df['Event 2'][i][j]].vector
		y = y/len(df['Event 2'][i])

		# Concatenate the two events and store
		x = np.concatenate((x, y))
		# df['Event 1'][i] = x
		# print(np.array(dfTToken['Event 1'][i]).shape)
		data[i] = np.array(x).reshape((1,-1))

	np.savetxt('../glove_Averaged_Expanded_Test.csv', data, delimiter = ',')

	return data, df


	



def tokenizeE1E2AndStore(df):
	tokenizer = RegexpTokenizer(r'\w(?:[-\w]*\w)?')
	df['Event 1'] = df['Event 1'].apply(tokenizer.tokenize)
	df['Event 2'] = df['Event 2'].apply(tokenizer.tokenize)

	
	for i in range(0, len(df['Event 1'])):
  		df['Event 1'][i] = [w for w in df['Event 1'][i] if not w in stop]

	for i in range(0, len(df['Event 2'])):
  		df['Event 2'][i] = [w for w in df['Event 2'][i] if not w in stop]
	return df



def main():

	# Loading Training Data
	dfRaw = pd.read_csv("test_expanded.csv", usecols = fields)
	dfTToken = tokenizeE1E2AndStore(dfRaw)
	print('Training Set Tokenized')
	trainData, dfTrain = embedAndConcatenate(dfTToken)
	print('Train set vectors formed, concatenated and stored')

	# Load Test Data
	# dfTestRaw = pd.read_csv("test_expanded.csv", usecols = fields)
	# print('Test set loaded')
	# dfTestToken = tokenizeE1E2AndStore(dfTestRaw)
	# print('Test set tokenized')
	# testData,_ = embedAndConcatenate(dfTestToken)
	# print('Test set vectors formed and concatenated')


main()


# Remove Stop Words
# Try BERT