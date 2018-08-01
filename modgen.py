
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from tqdm import tqdm_notebook as tqdm


train = pd.read_csv("/Users/jeromydiaz/Desktop/all/train.csv")
test = pd.read_csv("/Users/jeromydiaz/Desktop/all/test.csv")


# For Train
y_train = train['Survived']
x_train = train.drop(['PassengerId','Survived','Name','Ticket','Cabin','Embarked'], axis=1)
encSex = LabelEncoder()
testEnc = encSex.fit_transform(x_train.Sex)
x_train['Sex'] = testEnc
x_train['Age'].fillna(x_train.Age.mean(), inplace = True)
x_train.head()


# For Test
x_test = test.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1)
x_test['Sex'] = encSex.fit_transform(x_test.Sex)
x_test['Age'].fillna(x_test.Age.mean(), inplace = True)
x_test.fillna(x_test[x_test.Pclass == 3].Fare.mean(), inplace = True)
x_test.head()


def aucFunction(y_given, pred_given):
    falsePositive, truePositive, threshold = roc_curve(y_given, pred_given)
    roc_auc = auc(falsePositive, truePositive)

    return roc_auc


def getRandomNumber(firstNumber, lastNumber, getType = 'int'):
    # Returns int between first and last number
    if getType == 'exp':
        if firstNumber > 0 and lastNumber > 0:
            expList = np.logspace(firstNumber, lastNumber, (abs(lastNumber) - abs(firstNumber) + 1))
        elif firstNumber < 0 and lastNumber < 0:
            expList = np.logspace(firstNumber, lastNumber, (abs(firstNumber) - abs(lastNumber) + 1))
        elif firstNumber == 0 or lastNumber == 0:
            print('Cannot use 0 as an argument - Use 1 instead')
        else:
            expList = np.logspace(firstNumber, lastNumber, (abs(firstNumber) + abs(lastNumber) + 1))
        return expList[np.random.randint(0, len(expList))]
    # Returns float between first and last number
    elif getType == 'float':
        return (lastNumber - firstNumber) * np.random.rand() + firstNumber
    else:
        return np.random.randint(firstNumber, lastNumber + 1)


def getRandomFromList(inputList):
    return inputList[getRandomNumber(0, len(inputList) - 1)]


def dataFrameUpdate(params, y_train, y_valid, pred_train, pred_valid, df):
    updateDF = df
    updateParams = params
    updateParams['Train Accuracy'] = accuracy_score(y_train, pred_train)
    updateParams['Train Loss'] = mean_absolute_error(y_train, pred_train)
    updateParams['Train Auc'] = aucFunction(y_train, pred_train)
    updateParams['Valid Accuracy'] = accuracy_score(y_valid, pred_valid)
    updateParams['Valid Loss'] = mean_absolute_error(y_valid, pred_valid)
    updateParams['Valid Auc'] = aucFunction(y_valid, pred_valid)
    s = pd.Series(updateParams)
    updateDF = updateDF.append(s, ignore_index=True)

    return updateDF


def getSavedParams(rowNum):
    analysisDF = pd.read_csv('//Users/jeromydiaz/Desktop/Titanic_AnalysisDF.csv')
    s = analysisDF.sort_values(['Valid Auc','Train Auc'], ascending = False).iloc[[rowNum]].T.squeeze()
    dropList = ['Unnamed: 0','Valid Accuracy', 'Valid Auc', 'Valid Loss', 'Train Accuracy',
                           'Train Auc', 'Train Loss', 'verbose', 'model_type']

    for columnName in dropList:
        if columnName in s.keys():
            s = s.drop(columnName)

    params = s.to_dict()
    # 'min_samples_split', 'min_samples_leaf' were changed because they are percentages - Check All Other
    toIntList = ['lambda_l1', 'lambda_l2', 'max_bin', 'max_depth', 'min_data', 'num_leaves', 'num_trees',
                'bagging_freq', 'n_estimators', 'degree', 'max_features', 'n_neighbors', 'p']

    for feature in toIntList:
        if feature in params:
            params[feature] = int(params[feature])

    toFloatList = ['gamma']
    for feature in toFloatList:
        if feature in params:
            params[feature] = float(params[feature])

    return params


def trainFullSubmission(pred_test, ID, ensemble, ensembleList):
    # Ensemble Models Together
    if ensemble:
        pred_test = np.where(np.array(ensembleList).mean(axis = 0) > 0.5, 1,0)
        print("Ensemble Predition: " + str(np.unique(pred_test, return_counts = True)))

    # Final Submission
    finaldf = pd.DataFrame(ID)
    finaldf['Survived'] = pred_test
    finaldf.set_index('PassengerId', inplace = True)
    finaldf.to_csv('/Users/jeromydiaz/Desktop/TitanicFinalSub.csv')


def scaleSelector(x_train, x_valid, x_test, Scaler):
    scaler = Scaler
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    return x_train, x_valid, x_test


def modelSelector(x_train, y_train, x_valid, x_test, train_full, Model, modeltype = None):
    if modeltype == 'lightgbm':
        model = Model
        pred_train = model.predict(x_train, num_iteration = model.best_iteration)
        pred_valid = model.predict(x_valid, num_iteration = model.best_iteration)
        if train_full: pred_test = model.predict(x_test, num_iteration = model.best_iteration)
    else:
        # Needs work for other models when getting to that point (Valid will be weird with train_full)
        model = Model
        model.fit(x_train, y_train)
        pred_train = model.predict(x_train)
        pred_valid = model.predict(x_valid)
        if train_full: pred_test = model.predict(x_test)
    pred_train = np.where(pred_train > 0.5, 1, 0)
    pred_valid = np.where(pred_valid > 0.5, 1, 0)
    if train_full:
        pred_test = np.where(pred_test > 0.5, 1, 0)
    else:
        pred_test = None

    return pred_train, pred_valid, pred_test


def getModelLasso(previousModel, rowNum = 0):
    # Lasso model will create more and more sparse coef by deciding which features are important and
    # ignorning the rest.  The term lasso means lassoing up all the good features and leaving out the bad ones
    if not previousModel:
        params = {}
        params['model_type'] = 'Lasso_classifier'
        params['alpha'] = getRandomNumber(-15,2, getType = 'exp')
    else:
        params = getSavedParams(rowNum)

    model = Lasso(alpha = params['alpha'], max_iter = 1e6, random_state = 0)

    return params, model


def getModelRidge(previousModel, rowNum = 0):
    # Ridge model used as L2 Regularization.  The model itself will keep the coef from blowing up or shrinking
    # down to a superlow or superhigh number causing massive overfitting of the data.  Generally alpha will be
    # between 1e-15 and 1e-3.  Can use all features at once, but computationally heavy as you add more features
    if not previousModel:
        params = {}
        params['model_type'] = 'Ridge_classifier'
        params['alpha'] = getRandomNumber(-15,2, getType = 'exp')
    else:
        params = getSavedParams(rowNum)

    model = Ridge(alpha = params['alpha'], max_iter = 1e6, random_state = 0)

    return params, model


def getModelKNN(previousModel, rowNum = 0):
    # K Nearest Neighbor Model
    if not previousModel:
        params = {}
        params['model_type'] = 'KNN_classifier'
        params['n_neighbors'] = getRandomNumber(1,30)
        params['p'] = getRandomNumber(1,2)

    else:
        params = getSavedParams(rowNum)

    model = KNeighborsClassifier(n_neighbors = params['n_neighbors'], p = params['p'])

    return params, model


def getModelGradientBoosting(previousModel, rowNum = 0):
    # Gradient Boosting Model
    if not previousModel:
        params = {}
        params['model_type'] = 'Gradient_boosting_classifier'
        params['learning_rate'] = getRandomNumber(0.001,1, getType = 'float')
        params['n_estimators'] = getRandomNumber(1,500)
        params['max_depth'] = getRandomNumber(1,32)
        params['min_samples_split'] = getRandomNumber(0.1,1, getType = 'float')
        params['min_samples_leaf'] = getRandomNumber(0.1,0.5, getType = 'float')
        params['max_features'] = getRandomNumber(1,X_train.shape[1])
    else:
        params = getSavedParams(rowNum)

    model = GradientBoostingClassifier(learning_rate = params['learning_rate'],
                                       n_estimators = params['n_estimators'],
                                       max_depth = params['max_depth'],
                                       min_samples_split = params['min_samples_split'],
                                       min_samples_leaf = params['min_samples_leaf'],
                                       max_features = params['max_features'],
                                       random_state = 0)
    return params, model


def makeModelSVC(previousModel, rowNum = 0):
    # Does not need a ton of randomizations (n=250) should be fine - Time consuming
    # MinMaxScaler() seems to work the best

    if not previousModel:
        params = {}
        params['model_type'] = 'Support_vector_classifier'
        params['kernel'] =  getRandomFromList(['linear', 'rbf', 'poly']) # Can only be one rbf/poly non-linear
        if params['kernel'] == 'rbf' or params['kernel'] == 'poly':
            params['gamma'] = getRandomNumber(0.001,5, getType = 'float') # For non-linear  only [0.1 - 100]
        else:
            params['gamma'] = 'auto'
        params['C'] = getRandomNumber(0.01,50, getType = 'float') # [0.1 - 1000]
        if params['kernel'] == 'poly':
            params['degree'] = getRandomNumber(1,4) # Only used on POLY - [1,2,3,4,5,6]
        else:
            params['degree'] = 3

    else:
        params = getSavedParams(rowNum)

    model = SVC(kernel = params['kernel'], gamma = params['gamma'], C = params['C'], degree = params['degree'],
               max_iter = 1e6, random_state = 0)

    return params, model


def makeModelAdaBoostTree(previousModel, rowNum = 0):
    if not previousModel:
        params = {}
        params['model_type'] = 'AdaBoost_with_decision_tree'
        params['learning_rate'] = getRandomNumber(0.001,1, getType = 'float')
        params['n_estimators'] = getRandomNumber(2,1000)
        params['max_depth'] = getRandomNumber(1,500)
        params['min_samples_split'] = getRandomNumber(0.1,1, getType = 'float')
        params['min_samples_leaf'] = getRandomNumber(0.1,0.5, getType = 'float')
    else:
        params = getSavedParams(rowNum)

    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth = params['max_depth'],
                                min_samples_split = params['min_samples_split'],
                                min_samples_leaf = params['min_samples_leaf'], random_state = 0),
                                learning_rate = params['learning_rate'], n_estimators = params['n_estimators'],
                                random_state = 0)
    return params, model


def makeModelDecisionTree(previousModel, rowNum = 0):
    if not previousModel:
        params = {}
        params['model_type'] = 'decision_tree'
        params['max_depth'] = getRandomNumber(1,50)
        params['min_samples_split'] = getRandomNumber(0.1,1, getType = 'float')
        params['min_samples_leaf'] = getRandomNumber(0,0.5, getType = 'float')
    else:
        params = getSavedParams(rowNum)

    model = DecisionTreeClassifier(max_depth = params['max_depth'],
                                   min_samples_split = params['min_samples_split'],
                                   min_samples_leaf = params['min_samples_leaf'], random_state = 0)
    return params, model


def makeModelRandomForest(previousModel, rowNum = 0):
        if not previousModel:
            params = {}
            params['model_type'] = 'random_forest'
            params['n_estimators'] = getRandomNumber(2,1000)
            params['max_depth'] = getRandomNumber(1,50)
            params['min_samples_split'] = getRandomNumber(0.1,1, getType = 'float')
            params['min_samples_leaf'] = getRandomNumber(0,0.5, getType = 'float')
        else:
            params = getSavedParams(rowNum)

        model = RandomForestClassifier(n_estimators = params['n_estimators'],
                                        max_depth = params['max_depth'],
                                        min_samples_split = params['min_samples_split'],
                                        min_samples_leaf = params['min_samples_leaf'], random_state = 0)
        return params, model


# Model Options
validation_set = True
scaler_select = True
ensemble = False
lightModel = False

if validation_set:
    X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, random_state = 5)
    X_test = x_test
else:
    X_train, Y_train = x_train, y_train
    X_test = x_test
    Y_test = None

# Scaler Function - Can Call StandardScaler(), Normalizer(), MinMaxScaler(), RobustScaler()
if scaler_select: X_train, X_valid, X_test = scaleSelector(X_train, X_valid, X_test, StandardScaler())

# Init Lists
ensembleList = []
analysisDF = pd.DataFrame()

# Model Function:
#      Linear: LinearRegression(), LogisticRegression(), Perceptron(), Ridge(), Lasso()
#      Ensemble: RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()
#      Tree: DecisionTreeClassifier()
#      Neighbors: KNeighborsClassifier()
#      SVM: SVC()

# Parameter to Model
np.random.seed()

modelCreation = {
                    'lightgbm'     : 2500,
                    'lasso'        : 200,
                    'ridge'        : 200,
                    'knn'          : 200,
                    'gradboost'    : 500,
                    'svc'          : 250,
                    'adaboost'     : 500,
                    'decisiontree' : 500,
                    'randomforest' : 500
                }

numModels = 1
previousModel = False
train_full = False

for _ in tqdm(range(0,numModels)):

    if modelSelection == 'lightgbm':
        lgb_train = lgb.Dataset(X_train, label=Y_train)
        lgb_valid = lgb.Dataset(X_valid, Y_valid, reference = lgb_train)
        if not previousModel:
            num_trees = 10000
            params = {}
            params['learning_rate'] = getRandomNumber(0.01,1, getType = 'float')
            params['boosting_type'] = 'dart'
            params['objective'] = 'binary'
            params['metric'] = ['auc','binary_logloss']
            params['sub_feature'] = 0.5
            params['num_leaves'] = getRandomNumber(2,400)
            params['min_data'] = getRandomNumber(2,100)
            params['max_depth'] = getRandomNumber(1,200)
            params['lambda_l2'] = getRandomNumber(0,1)
            params['feature_fraction'] = getRandomNumber(0.5,1, getType = 'float')
            params['bagging_fraction'] = getRandomNumber(0.5,1, getType = 'float')
            params['bagging_freq'] = getRandomNumber(1,10)
            params['verbose'] = 0
            model = lgb.train(params, lgb_train, num_trees, valid_sets = lgb_valid, early_stopping_rounds = 50,
                             verbose_eval = False)
            params['num_trees'] = num_trees
            params['model_type'] = 'lightgbm_' + params['boosting_type']

        else:
            params = getSavedParams(0)
            num_trees = params['num_trees']
            params['metric'] = ['auc','binary_logloss']
            params.pop('num_trees', None)
            model = lgb.train(params,lgb_train,num_trees, valid_sets = lgb_valid, early_stopping_rounds = 20,
                             verbose_eval = False)

    elif modelSelection == 'lasso':
        # Lasso Model Generator
        params, model = getModelLasso(previousModel, rowNum = 0)

    elif modelSelection == 'ridge':
        # Ridge Model Generator
        params, model = getModelLasso(previousModel, rowNum = 0)

    elif modelSelection == 'knn':
        # KNN Model Generator
        params, model = getModelKNN(previousModel, rowNum = 0)

    elif modelSelection == 'gradboost':
        # Gradient Boosting Generator
        params, model = getModelGradientBoosting(previousModel, rowNum = 0)

    elif modelSelection == 'svc':
        # SVC Model Generator
        params, model = makeModelSVC(previousModel, rowNum = 0)

    elif modelSelection == 'adaboost':
        # AdaBoost with DecisionTreeClassifier Model Generator
        params, model = makeModelAdaBoostTree(previousModel, rowNum = 0)

    elif modelSelection == 'decisiontree':
        # DecisionTreeClassifier Model Generator
        params, model = makeModelDecisionTree(previousModel, rowNum = 0)

    elif modelSelection == 'randomforest':
        # RandomForest Model Generator
        params, model = makeModelRandomForest(previousModel, rowNum = 0)


    # Model Generation based off paramList and modelList
    Pred_train, Pred_valid, Pred_test = modelSelector(X_train, Y_train, X_valid, X_test, train_full,
                                                      model, modeltype = modelSelection)
    if not train_full:
        analysisDF = dataFrameUpdate(params, Y_train, Y_valid, Pred_train, Pred_valid, analysisDF)
    if ensemble: ensembleList.append(Pred_test)

if not train_full:
    print(analysisDF['Train Auc'].max(), analysisDF['Valid Auc'].max())
    plt.plot(range(0, numModels), analysisDF['Train Auc'], 'b', label = 'Train Auc')
    plt.plot(range(0, numModels), analysisDF['Valid Auc'], 'r', label = 'Valid Auc')
    plt.show()
if not previousModel:
    analysisDF = analysisDF.sort_values(['Valid Auc','Train Auc'], ascending = False)
    analysisDF.to_csv('//Users/jeromydiaz/Desktop/Titanic_AnalysisDF.csv')

# Writes final submission file
if train_full: trainFullSubmission(Pred_test, test['PassengerId'], ensemble, ensembleList)


analysisDF.sort_values(['Valid Auc','Train Auc'], ascending = False).head(10)
