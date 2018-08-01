
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
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
test_x = test.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1)
test_x['Sex'] = encSex.fit_transform(test_x.Sex)
test_x['Age'].fillna(test_x.Age.mean(), inplace = True)
test_x.fillna(test_x[test_x.Pclass == 3].Fare.mean(), inplace = True)
test_x.head()


def aucFunction(y_test, y_pred):
    falsePositive, truePositive, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(falsePositive, truePositive)
    return roc_auc


def getRandomNumber(firstNumber, lastNumber, getInt = True):
    # Returns int between first and last number
    if getInt:
        return np.random.randint(firstNumber, lastNumber + 1)
    # Returns float between first and last number
    else:
        return (lastNumber - firstNumber) * np.random.rand() + firstNumber


def dataFrameUpdate(params, y_train, y_test, train_pred, y_pred, df):
    updateDF = df
    updateParams = params
    updateParams['Train Accuracy'] = accuracy_score(y_train, train_pred)
    updateParams['Train Loss'] = mean_absolute_error(y_train, train_pred)
    updateParams['Train Auc'] = aucFunction(y_train, train_pred)
    updateParams['Test Accuracy'] = accuracy_score(y_test, y_pred)
    updateParams['Test Loss'] = mean_absolute_error(y_test, y_pred)
    updateParams['Test Auc'] = aucFunction(y_test, y_pred)
    s = pd.Series(updateParams)
    updateDF = updateDF.append(s, ignore_index=True)

    return updateDF


def getSavedParams(rowNum):
    analysisDF = pd.read_csv('//Users/jeromydiaz/Desktop/Titanic_AnalysisDF.csv')
    s = analysisDF.sort_values(['Test Auc','Train Auc'], ascending = False).iloc[[rowNum]].T.squeeze()
    s = s.drop(['Unnamed: 0','Test Accuracy', 'Test Auc', 'Test Loss', 'Train Accuracy',
                           'Train Auc', 'Train Loss', 'verbose', 'model_type'])
    params = s.to_dict()
    toIntList = ['lambda_l1', 'lambda_l2', 'max_bin', 'max_depth', 'min_data', 'num_leaves', 'num_trees',
                'bagging_freq']

    for feature in toIntList:
        if feature in params:
            params[feature] = int(params[feature])
    return params


def trainFullSubmission(y_pred, ID, ensemble, ensembleList):
    # Ensemble Models Together
    if ensemble:
        y_pred = np.where(np.array(ensembleList).mean(axis = 0) > 0.5, 1,0)
        print("Ensemble Predition: " + str(np.unique(y_pred, return_counts = True)))

    # Final Submission
    finaldf = pd.DataFrame(ID)
    finaldf['Survived'] = y_pred
    finaldf.set_index('PassengerId', inplace = True)
    finaldf.to_csv('/Users/jeromydiaz/Desktop/TitanicFinalSub.csv')


def scaleSelector(x_train, x_test, Scaler):
    scaler = Scaler
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def modelSelector(x_train, y_train, x_test, y_test, train_full, Model, modeltype = None):
    if modeltype == 'lgb':
        model = Model
        train_pred = model.predict(x_train, num_iteration = model.best_iteration)
        y_pred = model.predict(x_test, num_iteration = model.best_iteration)
    else:
        model = Model
        model.fit(x_train, y_train)
    # train_pred = model.predict(x_train)
    train_pred = np.where(train_pred > 0.5, 1, 0)
    # y_pred = model.predict(x_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return train_pred, y_pred


# Model Options
train_full = False
scaler_select = True
ensemble = False
lightModel = True

if not train_full:
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, random_state = 5)
else:
    X_train, Y_train = x_train, y_train
    X_test = test_x
    Y_test = None

# Scaler Function - Can Call StandardScaler(), Normalizer(), MinMaxScaler(), RobustScaler()
if scaler_select: X_train, X_test = scaleSelector(X_train, X_test, RobustScaler())

# Init Lists
ensembleList = []
modelList = []
cachesParams = []
bestTrees = []
analysisDF = pd.DataFrame()

# Model Function:
#      Linear: LinearRegression(), LogisticRegression(), Perceptron(), Ridge(), Lasso()
#      Ensemble: RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()
#      Tree: DecisionTreeClassifier()
#      SVM: SVC()

# Parameter to Model
np.random.seed(12)

numModels = 1000
previousModel = False

for _ in tqdm(range(0,numModels)):
    if lightModel:
        lgb_train = lgb.Dataset(X_train, label=Y_train)
        lgb_valid = lgb.Dataset(X_test, Y_test, reference = lgb_train)
        if not previousModel:
            num_trees = getRandomNumber(2,5000)
            params = {}
            params['learning_rate'] = getRandomNumber(0.01,1, getInt = False)
            params['boosting_type'] = 'gbdt'
            params['objective'] = 'binary'
            params['metric'] = ['auc','binary_logloss']
            params['sub_feature'] = 0.5
            params['num_leaves'] = getRandomNumber(2,400)
            params['min_data'] = getRandomNumber(2,100)
            params['max_depth'] = getRandomNumber(1,200)
            # params['max_bin'] = getRandomNumber(2,100)
            # params['lambda_l1'] = 0
            params['lambda_l2'] = getRandomNumber(0,1)
            params['feature_fraction'] = getRandomNumber(0.5,1, getInt = False) # 0.9
            params['bagging_fraction'] = getRandomNumber(0.5,1, getInt = False) # 0.8
            params['bagging_freq'] = getRandomNumber(1,10)
            params['verbose'] = 0
            model = lgb.train(params,lgb_train,num_trees, valid_sets = lgb_valid, early_stopping_rounds = 20,
                             verbose_eval = False)
            params['num_trees'] = num_trees
            params['model_type'] = 'lightgbm_' + params['boosting_type']
            bestTrees.append(model.best_iteration)

        else:
            params = getSavedParams(0)
            num_trees = params['num_trees']
            params.pop('num_trees', None)
            model = lgb.train(params,lgb_train,num_trees)

    else:
        modelList.append()

    # Model Generation based off paramList and modelList
    Train_pred, Y_pred = modelSelector(X_train, Y_train, X_test, Y_test, train_full, model, modeltype = 'lgb')
    if not train_full:
        analysisDF = dataFrameUpdate(params, Y_train, Y_test, Train_pred, Y_pred, analysisDF)
    if ensemble: ensembleList.append(Y_pred)

if not train_full:
    print(analysisDF['Train Auc'].max(), analysisDF['Test Auc'].max())
    plt.plot(range(0, numModels), analysisDF['Train Auc'], 'b', label = 'Auc Train')
    plt.plot(range(0, numModels), analysisDF['Test Auc'], 'r', label = 'Auc Test')
    plt.show()
if not previousModel:
    analysisDF = analysisDF.sort_values(['Test Auc','Train Auc'], ascending = False)
    analysisDF.to_csv('//Users/jeromydiaz/Desktop/Titanic_AnalysisDF.csv')

# Writes final submission file
if train_full: trainFullSubmission(Y_pred, test['PassengerId'], ensemble, ensembleList)


analysisDF.sort_values(['Test Auc','Train Auc'], ascending = False).head(10)
