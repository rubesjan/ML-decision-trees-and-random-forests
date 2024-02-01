# Machile learning - implementation of decision trees and random forests

Implementation of machine learning algorithms for predicting binary values - decision trees (ID3) and bagging - random forests

The id3 module implements MyDecisionTree class, that has fit and predict methods for learning and predicting y binary values.<br>
MyDecisionTree class is implemented in id3/model_class.py

The forest module implements random forest classifier, that also has fit and predict methods for learning and predicting y binary values.<br>
MyForest class is implemented in forest/forest_class.py

The modules uses pandas (DataFrame and Series) and numpy (predictions are returned in numpy array)

to run tests use command:<br>
pytest ./tests/test.py
