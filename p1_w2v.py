#  Required imports
import pandas as pd
import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn import svm



fields = ['Event 1', 'Event 2', 'Label']


def trainAndTestBaseline(trainData, testData, dfTrain, dfTest):

	# print(dfTrain['Event 1'][1].shape)

	causal_classify = svm.SVC(kernel = 'linear')
	_ = causal_classify.fit(trainData, dfTrain['Label'])
	print('Training Complete')

	# Load and Process Test Data
	prediction_result = causal_classify.predict(testData)
	print('Result')
	print(round(100*np.mean(prediction_result == dfTest['Label']), 2))
	# print(confusion_matrix(dfTestRaw['Label'], prediction_result))
	# for i in range(0, prediction_result.shape[0]):
	# 	print(str(prediction_result[i]) + " " + str(dfTestToken['Label'][i]))

	return prediction_result






def main():
	# Loading Training Data
	dfTrainRaw = pd.read_csv("train_expanded.csv", usecols = fields)
	dfTestRaw = pd.read_csv("P1_testing_set.csv", usecols = fields)
	trainData = np.loadtxt('../w2v_Averaged_Expanded_Train.csv', delimiter = ',')
	testData = np.loadtxt('../w2v_Averaged_Expanded_Test.csv', delimiter = ',')
	# print(trainData.shape, testData.shape)
	# print('Train set vectors formed, concatenated and stored')
	# print(trainData.shape)
	print('Proceeding for Training')
	prediction_result = trainAndTestBaseline(trainData, testData, dfTrainRaw, dfTestRaw)


main()


# Remove Stop Words
# Try BERT