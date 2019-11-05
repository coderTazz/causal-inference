#  Required imports
import pandas as pd
import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score



fields = ['Event 1', 'Event 2', 'Label']


def trainAndTestBaseline(trainData, testData, dfTrain, dfTest):


	causal_classify = svm.SVC(kernel = 'linear')
	_ = causal_classify.fit(trainData, dfTrain['Label'])
	print('Training Complete')

	# Load and Process Test Data
	prediction_result = causal_classify.predict(testData)
	print()
	print('Result')
	print("Accuracy: " + str(round(100*np.mean(prediction_result == dfTest['Label']), 2)) + "%")


	cm = confusion_matrix(prediction_result, dfTest['Label'])
	print("Confusion Matrix: ")
	print(cm)

	return prediction_result






def main():
	# Loading Training Data
	dfTrainRaw = pd.read_csv("train_expanded.csv", usecols = fields)
	dfTestRaw = pd.read_csv("test_expanded.csv", usecols = fields)
	trainData = np.loadtxt('../w2v_Averaged_Expanded_Train.csv', delimiter = ',')
	testData = np.loadtxt('../w2v_Averaged_Expanded_Test.csv', delimiter = ',')
	print('Proceeding for Training')
	prediction_result = trainAndTestBaseline(trainData, testData, dfTrainRaw, dfTestRaw)


main()


# Remove Stop Words
# Try BERT