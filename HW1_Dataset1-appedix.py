# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:35:52 2018

@author: HP
"""

# Cross validation check by varing validation size

validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(ScaledX, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('TREE', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Boosting', AdaBoostClassifier()))
models.append(('ANN', MLPClassifier()))


# evaluate each model in turn
results = []
names = []
n_splits = 10
for name, model in models:
	kfold = model_selection.KFold(n_splits, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

filehandler = open('cvResults0.3.obj','wb')
pickle.dump(results, filehandler)
#test1 = pickle.load(open("cvResults0.3.obj","rb"))




validation_size = 0.40
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(ScaledX, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('TREE', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Boosting', AdaBoostClassifier()))
models.append(('ANN', MLPClassifier()))


# evaluate each model in turn
results = []
names = []
n_splits = 10
for name, model in models:
	kfold = model_selection.KFold(n_splits, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

filehandler = open('cvResults0.4.obj','wb')
pickle.dump(results, filehandler)
filehandler.close()




validation_size = 0.50
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(ScaledX, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('TREE', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Boosting', AdaBoostClassifier()))
models.append(('ANN', MLPClassifier()))


# evaluate each model in turn
results = []
names = []
n_splits = 10
for name, model in models:
	kfold = model_selection.KFold(n_splits, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

filehandler = open('cvResults0.5.obj','wb')
pickle.dump(results, filehandler)
filehandler.close()


validation_size = 0.60
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(ScaledX, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('TREE', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Boosting', AdaBoostClassifier()))
models.append(('ANN', MLPClassifier()))


# evaluate each model in turn
results = []
names = []
n_splits = 10
for name, model in models:
	kfold = model_selection.KFold(n_splits, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

filehandler = open('cvResults0.6.obj','wb')
pickle.dump(results, filehandler)
filehandler.close()




validation_size = 0.70
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(ScaledX, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('TREE', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Boosting', AdaBoostClassifier()))
models.append(('ANN', MLPClassifier()))


# evaluate each model in turn
results = []
names = []
n_splits = 10
for name, model in models:
	kfold = model_selection.KFold(n_splits, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

filehandler = open('cvResults0.7.obj','wb')
pickle.dump(results, filehandler)
filehandler.close()




validation_size = 0.80
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(ScaledX, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('TREE', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Boosting', AdaBoostClassifier()))
models.append(('ANN', MLPClassifier()))


# evaluate each model in turn
results = []
names = []
n_splits = 10
for name, model in models:
	kfold = model_selection.KFold(n_splits, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

filehandler = open('cvResults0.8.obj','wb')
pickle.dump(results, filehandler)
filehandler.close()
