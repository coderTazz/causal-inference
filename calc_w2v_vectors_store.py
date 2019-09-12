#  Required imports
import pandas as pd
# import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import svm



fields = ['Event 1', 'Event 2', 'Label']


def trainAndTestBaseline(trainData, dfTrain):

	# print(dfTrain['Event 1'][1].shape)

	causal_classify = svm.SVC(kernel = 'poly', degree = 4, decision_function_shape = 'ovr')
	_ = causal_classify.fit(trainData, dfTrain['Label'])
	print('Training Complete')

	# Load and Process Test Data
	dfTestRaw = pd.read_csv("P1_testing_set.csv", usecols = fields)
	print('Test set loaded')
	dfTestToken = tokenizeE1E2AndStore(dfTestRaw, False)
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


	# Load spacy model
	nlp = spacy.load('en_core_web_lg')

	data = np.zeros((df.shape[0], 2*len(nlp.vocab['dog'].vector)))
	# print(trainData.shape)

	for i in range(0, df.shape[0]):
		
		# Event1
		x = np.zeros(len(nlp.vocab['dog'].vector,))
		for j in range(0, len(df['Event 1'].iloc(0)[i])):
			x = x + nlp.vocab[df['Event 1'].iloc(0)[i][j]].vector
		x = x/len(df['Event 1'].iloc(0)[i])

		# Event2
		y = np.zeros(len(nlp.vocab['dog'].vector,))
		for j in range(0, len(df['Event 2'].iloc(0)[i])):
			y = y + nlp.vocab[df['Event 2'].iloc(0)[i][j]].vector
		y = y/len(df['Event 2'].iloc(0)[i])

		# Concatenate the two events and store
		x = np.concatenate((x, y))
		# df['Event 1'][i] = x
		# print(np.array(dfTToken['Event 1'][i]).shape)
		data[i] = np.array(x).reshape((1,-1))

	return data, df

	



def tokenizeE1E2AndStore(dfRaw, storeBool):
	tokenizer = RegexpTokenizer(r'\w(?:[-\w]*\w)?')
	dfRaw['Event 1'] = dfRaw['Event 1'].apply(tokenizer.tokenize)
	dfRaw['Event 2'] = dfRaw['Event 2'].apply(tokenizer.tokenize)
	if storeBool:
		dfRaw.to_csv('Tokenized_Small_Train_Expanded_Set_W2V.csv')
	else:
		return dfRaw



def main():
	# Loading Training Data
	dfRaw = pd.read_csv("train_expanded.csv", usecols = fields)
	tokenizeE1E2AndStore(dfRaw, True)
	print('Training Set Tokenized')
	dfTToken = pd.read_csv("Tokenized_Small_Train_Expanded_Set_W2V.csv", usecols = fields)
	trainData, dfTrain = embedAndConcatenate(dfTToken)
	print('Train set vectors formed and concatenated')
	# print(trainData.shape)
	# print('Proceeding for Training')
	# prediction_result = trainAndTestBaseline(trainData, dfTrain)


main()


# Remove Stop Words
# Try BERT