#  Required imports
import pandas as pd
# import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
# import sklearn
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# from sklearn import svm
import gensim
from gensim.models import KeyedVectors



fields = ['Event 1', 'Event 2', 'Label']


def trainAndTestBaseline(trainData, dfTrain):

	# print(dfTrain['Event 1'][1].shape)

	causal_classify = svm.SVC(kernel = 'poly', degree = 4, decision_function_shape = 'ovr')
	_ = causal_classify.fit(trainData, dfTrain['Label'])
	print('Training Complete')

	# Load and Process Test Data
	dfTestRaw = pd.read_csv("P1_testing_set.csv", usecols = fields)
	print('Test set loaded')
	dfTestToken = tokenizeE1E2AndStore(dfTestRaw)
	print('Test set tokenized')
	testData,_ = embedAndConcatenate(dfTestToken)
	print('Test set vectors formed and concatenated')
	prediction_result = causal_classify.predict(testData)
	print('Result')
	print(round(100*np.mean(prediction_result == dfTestRaw['Label']), 2))
	print(confusion_matrix(dfTestRaw['Label'], prediction_result))
	# for i in range(0, prediction_result.shape[0]):
	# 	print(str(prediction_result[i]) + " " + str(dfTestToken['Label'][i]))

	return prediction_result


def embedAndConcatenate(df):


	# Load w2v model
	nlp = KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

	data = np.zeros((df.shape[0], 2*(nlp['dog'].shape[0])))
	# print(trainData.shape)

	for i in range(0, df.shape[0]):
		
		# Event1
		x = np.zeros((nlp['dog'].shape[0],))
		n = len(df['Event 1'][i])
		for j in range(0, len(df['Event 1'][i])):
			if df['Event 1'][i][j] in nlp.vocab:
				x = x + nlp[df['Event 1'][i][j]]
			else:
				n = n - 1
		x = x/n

		# Event2
		y = np.zeros((nlp['dog'].shape[0],))
		n = len(df['Event 2'][i])
		for j in range(0, len(df['Event 2'][i])):
			if df['Event 2'][i][j] in nlp.vocab:
				x = x + nlp[df['Event 2'][i][j]]
			else:
				n = n - 1
		y = y/n

		# Concatenate the two events and store
		x = np.concatenate((x, y))
		# df['Event 1'][i] = x
		# print(np.array(dfTToken['Event 1'][i]).shape)
		data[i] = np.array(x).reshape((1,-1))

	np.savetxt('../w2v_Averaged_Expanded_Test.csv', data, delimiter = ',')

	return data, df

	



def tokenizeE1E2AndStore(df):
	tokenizer = RegexpTokenizer(r'\w(?:[-\w]*\w)?')
	df['Event 1'] = df['Event 1'].apply(tokenizer.tokenize)
	df['Event 2'] = df['Event 2'].apply(tokenizer.tokenize)
	
	return df



def main():

	# Loading Training Data
	dfRaw = pd.read_csv("train_expanded.csv", usecols = fields)
	dfTToken = tokenizeE1E2AndStore(dfRaw)
	print('Training Set Tokenized')
	# dfTToken = pd.read_csv("Tokenized_Small_Train_Expanded_Set_W2V.csv", usecols = fields)
	trainData, dfTrain = embedAndConcatenate(dfTToken)
	print('Train set vectors formed, concatenated and stored')
	# print(trainData.shape)
	# print('Proceeding for Training')
	# prediction_result = trainAndTestBaseline(trainData, dfTrain)

	# Load and Process Test Data
	dfTestRaw = pd.read_csv("P1_testing_set.csv", usecols = fields)
	print('Test set loaded')
	dfTestToken = tokenizeE1E2AndStore(dfTestRaw)
	print('Test set tokenized')
	testData,_ = embedAndConcatenate(dfTestToken)
	print('Test set vectors formed and concatenated')


main()


# Remove Stop Words
# Try BERT