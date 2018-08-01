# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


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


def seriesMeanSTD(df):
    index_list = df.count().keys()
    df_mean = df.mean().get_values()
    df_std = df.std().get_values()
    comb_mean_std = []

    for v1, v2 in zip(df_mean, df_std):
        comb_mean_std.append(("%.6f" % v1) + ' - ' + ("%.6f" % v2))

    return pd.Series(comb_mean_std, index_list)


def getRandomNumber(firstNumber, lastNumber, getType = 'int'):
    # Returns int between first and last number
    if getType == 'exp' or getType == 'exp_random':
        if firstNumber > 0 and lastNumber > 0:
            expList = np.logspace(firstNumber, lastNumber, (abs(lastNumber) - abs(firstNumber) + 1))
        elif firstNumber < 0 and lastNumber < 0:
            expList = np.logspace(firstNumber, lastNumber, (abs(firstNumber) - abs(lastNumber) + 1))
        elif firstNumber == 0 or lastNumber == 0:
            print('Cannot use 0 as an argument - Use 1 instead')
        else:
            expList = np.logspace(firstNumber, lastNumber, (abs(firstNumber) + abs(lastNumber) + 1))
        if getType == 'exp_random': expList = expList * np.random.randint(1, 9 + 1)
        return expList[np.random.randint(0, len(expList))]
    # Returns float between first and last number
    elif getType == 'float':
        return (lastNumber - firstNumber) * np.random.rand() + firstNumber
    else:
        return np.random.randint(firstNumber, lastNumber + 1)


def getRandomFromList(inputList):
    return inputList[getRandomNumber(0, len(inputList) - 1)]


def getSavedParams(load_index):
    analysisDF = pd.read_csv('//Users/jeromydiaz/Desktop/Titanic_AnalysisDF.csv')
    s = analysisDF[analysisDF['Unnamed: 0'] == load_index].dropna(axis = 1).T.squeeze()
    dropList = ['Unnamed: 0','Valid Accuracy', 'Valid Auc', 'Valid Loss', 'Train Accuracy',
                           'Train Auc', 'Train Loss', 'verbose', 'Model_type']

    modelSplit = s['Model_type'].split('_')

    if len(modelSplit) > 1:
        modelSelection = modelSplit[0]
        modelSelectionMod = modelSplit[1]
    else:
        modelSelection = modelSplit[0]
        modelSelectionMod = None

    for columnName in dropList:
        if columnName in s.keys():
            s = s.drop(columnName)

    params = s.to_dict()
    toIntList = ['max_bin', 'max_depth', 'min_data', 'num_leaves', 'num_trees',
                'bagging_freq', 'n_estimators', 'degree', 'max_features', 'n_neighbors', 'p', 'min_child_weight',
                'max_leaf_nodes', 'min_child_weight', 'seed', 'min_child_samples', 'subsample_freq']

    for feature in toIntList:
        if feature in params:
            params[feature] = int(params[feature])

    toFloatList = ['gamma']
    for feature in toFloatList:
        if feature in params:
            params[feature] = float(params[feature])

    return params, modelSelection, modelSelectionMod


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


def scaleSelector(x_train, x_test, scaler):
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test


def getModelLasso(previousModel, params):
    # Lasso model will create more and more sparse coef by deciding which features are important and
    # ignorning the rest.  The term lasso means lassoing up all the good features and leaving out the bad ones
    if not previousModel:
        params = {}
        params['Model_type'] = 'lasso'
        params['alpha'] = getRandomNumber(-15,2, getType = 'exp')

    model = Lasso(alpha = params['alpha'], max_iter = 1e6, random_state = 0)

    return params, model


def getModelRidge(previousModel, params):
    # Ridge model used as L2 Regularization.  The model itself will keep the coef from blowing up or shrinking
    # down to a superlow or superhigh number causing massive overfitting of the data.  Generally alpha will be
    # between 1e-15 and 1e-3.  Can use all features at once, but computationally heavy as you add more features
    if not previousModel:
        params = {}
        params['Model_type'] = 'ridge'
        params['alpha'] = getRandomNumber(-15,2, getType = 'exp')

    model = Ridge(alpha = params['alpha'], max_iter = 1e6, random_state = 0)

    return params, model


def getModelKNN(previousModel, params):
    # K Nearest Neighbor Model
    if not previousModel:
        params = {}
        params['Model_type'] = 'knn'
        params['n_neighbors'] = getRandomNumber(1,30)
        params['p'] = getRandomNumber(1,2)

    model = KNeighborsClassifier(n_neighbors = params['n_neighbors'], p = params['p'])

    return params, model


def getModelGradientBoosting(previousModel, params):
    # Gradient Boosting Model
    if not previousModel:
        params = {}
        params['Model_type'] = 'gradboost'
        params['learning_rate'] = getRandomNumber(-3,2, getType = 'exp_random')
        params['n_estimators'] = getRandomNumber(1,500)
        params['max_depth'] = getRandomNumber(1,32)
        params['min_samples_split'] = getRandomNumber(0.1,1, getType = 'float')
        params['min_samples_leaf'] = getRandomNumber(0.1,0.5, getType = 'float')
        params['max_features'] = getRandomNumber(1,X_train.shape[1])

    model = GradientBoostingClassifier(learning_rate = params['learning_rate'],
                                       n_estimators = params['n_estimators'],
                                       max_depth = params['max_depth'],
                                       min_samples_split = params['min_samples_split'],
                                       min_samples_leaf = params['min_samples_leaf'],
                                       max_features = params['max_features'],
                                       random_state = 0)
    return params, model


def getModelSVC(previousModel,params):
    # Does not need a ton of randomizations (n=250) should be fine - Time consuming
    # MinMaxScaler() seems to work the best

    if not previousModel:
        params = {}
        params['Model_type'] = 'svc'
        params['kernel'] =  getRandomFromList(['linear', 'rbf', 'poly']) # Can only be one rbf/poly non-linear
        if params['kernel'] == 'rbf' or params['kernel'] == 'poly':
            params['gamma'] = getRandomNumber(-3,2, getType = 'exp_random') # For non-linear  only [0.1 - 100]
        else:
            params['gamma'] = 'auto'
        params['C'] = getRandomNumber(-2,2, getType = 'exp_random') # [0.1 - 1000]
        if params['kernel'] == 'poly':
            params['degree'] = getRandomNumber(1,4) # Only used on POLY - [1,2,3,4,5,6]
        else:
            params['degree'] = 3


    model = SVC(kernel = params['kernel'], gamma = params['gamma'], C = params['C'], degree = params['degree'],
               max_iter = 1e6, random_state = 0)

    return params, model


def getModelAdaBoostTree(previousModel, params):
    if not previousModel:
        params = {}
        params['Model_type'] = 'adaboost'
        params['learning_rate'] = getRandomNumber(-3,1, getType = 'exp_random')
        params['n_estimators'] = getRandomNumber(2,300)
        params['max_depth'] = getRandomNumber(1,100)
        params['min_samples_split'] = getRandomNumber(0.1,1, getType = 'float')
        params['min_samples_leaf'] = getRandomNumber(0.1,0.5, getType = 'float')

    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth = params['max_depth'],
                                min_samples_split = params['min_samples_split'],
                                min_samples_leaf = params['min_samples_leaf'], random_state = 0),
                                learning_rate = params['learning_rate'], n_estimators = params['n_estimators'],
                                random_state = 0)
    return params, model


def getModelDecisionTree(previousModel, params):
    if not previousModel:
        params = {}
        params['Model_type'] = 'decisiontree'
        params['max_depth'] = getRandomNumber(1,50)
        params['min_samples_split'] = getRandomNumber(0.1,1, getType = 'float')
        params['min_samples_leaf'] = getRandomNumber(0,0.5, getType = 'float')

    model = DecisionTreeClassifier(max_depth = params['max_depth'],
                                   min_samples_split = params['min_samples_split'],
                                   min_samples_leaf = params['min_samples_leaf'], random_state = 0)
    return params, model


def getModelRandomForest(previousModel, params):
    if not previousModel:
        params = {}
        params['Model_type'] = 'randomforest'
        params['n_estimators'] = getRandomNumber(2,500)
        params['max_depth'] = getRandomNumber(1,50)
        params['min_samples_split'] = getRandomNumber(0.1,1, getType = 'float')
        params['min_samples_leaf'] = getRandomNumber(0,0.5, getType = 'float')

    model = RandomForestClassifier(n_estimators = params['n_estimators'],
                                    max_depth = params['max_depth'],
                                    min_samples_split = params['min_samples_split'],
                                    min_samples_leaf = params['min_samples_leaf'], random_state = 0)
    return params, model


def getModelXGBoost(previousModel, params):
    # If max_leaf_nodes is specified then max_depth is ignored.
    if not previousModel:
        params = {}
        params['Model_type'] = 'xgboost'
        params['learning_rate'] = getRandomNumber(-5,1, getType = 'exp_random')
        params['min_child_weight'] = getRandomNumber(1,10)
        params['max_depth'] = getRandomNumber(1,10)
        # params['max_leaf_nodes'] = getRandomNumber(1,100)
        params['gamma'] = getRandomNumber(-5,5, getType = 'exp')
        params['subsample'] = getRandomNumber(0.5,1, getType = 'float')
        params['colsample_bytree'] = getRandomNumber(0.5,1, getType = 'float')
        params['objective'] = 'binary:logistic'
        params['n_estimators'] = getRandomNumber(2,100)

    model = XGBClassifier(learning_rate = params['learning_rate'], min_child_weight = params['min_child_weight'],
                         max_depth = params['max_depth'], gamma = params['gamma'], subsample = params['subsample'],
                         colsample_bytree = params['colsample_bytree'], objective = params['objective'],
                         n_estimators = params['n_estimators'], random_state = 0)

    return params, model


def getModelLightGBM(previousModel, params):
    if not previousModel:
        num_trees = 10000
        params = {}
        params['boosting_type'] = getRandomFromList(['dart', 'gbdt', 'rf'])
        params['Model_type'] = 'lightgbm_' + params['boosting_type']
        params['learning_rate'] = getRandomNumber(-3,2, getType = 'exp_random')
        params['objective'] = 'binary'
        params['metric'] = ['auc','binary_logloss']
        params['num_leaves'] = getRandomNumber(2,400)
        params['min_child_samples'] = getRandomNumber(2,100)
        params['max_depth'] = getRandomNumber(1,200)
        params['reg_lambda'] = getRandomNumber(-9,3, getType = 'exp_random')
        params['colsample_bytree'] = getRandomNumber(0.5,1, getType = 'float')
        params['subsample'] = getRandomNumber(0.5,1, getType = 'float')
        params['subsample_freq'] = getRandomNumber(1,10)
        params['n_estimators'] = 10000

    model = LGBMClassifier(boosting_type = params['boosting_type'], num_leaves = params['num_leaves'],
                          max_depth = params['max_depth'], learning_rate = params['learning_rate'],
                          n_estimators = params['n_estimators'], objective = params['objective'],
                          reg_lambda = params['reg_lambda'],colsample_bytree = params['colsample_bytree'],
                          min_child_samples = params['min_child_samples'], subsample = params['subsample'],
                          subsample_freq = params['subsample_freq'], random_state = 0)

    return params, model


def dataFrameUpdate(params, y_train, y_valid, pred_train, pred_valid, analysis_df, kfold_df, update_Kfold = False):
    updateDF = analysis_df
    updateParams = params
    if update_Kfold:
        updateParams['Train Accuracy'] = kfold_df['Train Accuracy']
        updateParams['Train Loss'] = kfold_df['Train Loss']
        updateParams['Train Auc'] = kfold_df['Train Auc']
        updateParams['Valid Accuracy'] = kfold_df['Valid Accuracy']
        updateParams['Valid Loss'] = kfold_df['Valid Loss']
        updateParams['Valid Auc'] = kfold_df['Valid Auc']
    else:
        updateParams['Train Accuracy'] = accuracy_score(y_train, pred_train)
        updateParams['Train Loss'] = mean_absolute_error(y_train, pred_train)
        updateParams['Train Auc'] = aucFunction(y_train, pred_train)
        updateParams['Valid Accuracy'] = accuracy_score(y_valid, pred_valid)
        updateParams['Valid Loss'] = mean_absolute_error(y_valid, pred_valid)
        updateParams['Valid Auc'] = aucFunction(y_valid, pred_valid)
    s = pd.Series(updateParams)
    updateDF = updateDF.append(s, ignore_index=True)

    return updateDF


def modelSelector(x_train, y_train, x_valid, y_valid, x_test, train_full, Model, kfold = 1, kfold_type = 'normal',
                  modeltype = None):
    # Determines if using KFold or single split
    if kfold > 1:
        kfold_DF = pd.DataFrame()
        kfold_pred_train = []
        kfold_pred_valid = []
        kfold_params = {}
        if kfold_type[0] == 'normal': kfold_gen = KFold(n_splits = kfold, random_state = 0)
        if kfold_type[0] == 'normal_repeat': kfold_gen = RepeatedKFold(n_splits = kfold, n_repeats = kfold_type[1],
                                                                       random_state = 0)
        if kfold_type[0] == 'strat': kfold_gen = StratifiedKFold(n_splits = kfold, random_state = 0)
        if kfold_type[0] == 'strat_repeat': kfold_gen = RepeatedStratifiedKFold(n_splits = kfold,
                                                                                n_repeats = kfold_type[1],
                                                                                random_state = 0)
        iteration_loop = kfold_gen.split(x_train, y_train)
    else:
        # iteration_loop is a throwaway to pass into for single split
        X_train, X_valid, Y_train, Y_valid = x_train, x_valid, y_train, y_valid
        iteration_loop = zip([None],[None])

    # Loop for Kfold or single split
    for train_index, valid_index in iteration_loop:
        if kfold > 1:
            X_train, X_valid = x_train[train_index], x_train[valid_index]
            Y_train, Y_valid = y_train[train_index], y_train[valid_index]

        if modeltype == 'lightgbm':
            # If you keep.. remember to change x_train to X_train (CAPS)
            model = Model
            model.fit(X_train, Y_train, eval_set = (X_valid, Y_valid), eval_metric = ['auc','binary_logloss'],
                      early_stopping_rounds = 40, verbose = False)
            pred_train = model.predict(X_train, num_iteration = model.best_iteration_)
            pred_valid = model.predict(X_valid, num_iteration = model.best_iteration_)
            if train_full: pred_test = model.predict(x_test, num_iteration = model.best_iteration_)
        else:
            model = Model
            model.fit(X_train, Y_train)
            pred_train = model.predict(X_train)
            pred_valid = model.predict(X_valid)
            if train_full: pred_test = model.predict(x_test)
        pred_train = np.where(pred_train > 0.5, 1, 0)
        pred_valid = np.where(pred_valid > 0.5, 1, 0)

        if kfold > 1:
            kfold_DF = dataFrameUpdate(kfold_params, Y_train, Y_valid, pred_train, pred_valid, kfold_DF,
                                       None)
            kfold_pred_train.append(pred_train)
            kfold_pred_valid.append(pred_valid)

    if train_full:
        pred_test = np.where(pred_test > 0.5, 1, 0)
    else:
        pred_test = None

    if kfold > 1:
        # pred_train = np.where(np.array(kfold_pred_train).mean(axis = 0) > 0.5, 1,0)
        # pred_valid = np.where(np.array(kfold_pred_valid).mean(axis = 0) > 0.5, 1,0)
        pred_train = None
        pred_valid = None
        kfold_mean_DF = kfold_DF.mean()
        kfold_mean_std_S = seriesMeanSTD(kfold_DF)
    else:
        kfold_mean_DF = None
        kfold_mean_std_S = None
    return pred_train, pred_valid, pred_test, kfold_mean_std_S


# Model Options
kfold_update = True
scaler_select = True
previousModel = False
train_full = False
ensemble = False

# Initialization of Lists and DFs
ensembleList = []
analysisDF = pd.DataFrame()
kfold_DF = pd.DataFrame()
params = {}
totalModels = 0

if previousModel:
    params, modelSelection, modelSelectionMod = getSavedParams(load_index = 3180)
    modelCreation = {modelSelection : 1}
else:
    modelCreation = {
                        'lightgbm'     : 2000,
                        'xgboost'      : 1500,
                        'lasso'        : 200,
                        'ridge'        : 200,
                        'knn'          : 200,
                        'gradboost'    : 500,
                        'svc'          : 250,
                        'decisiontree' : 500,
                        'randomforest' : 250
#                         'adaboost'     : 200
                    }

if previousModel: kfold_update = False
if kfold_update:
    kfold = 4
    kfold_type = ('normal', 1) # 'normal', 'strat', 'normal_repeat', 'strat_repeat' - (type, # repeats)
else: kfold = 1


# Scaler Function - Can Call StandardScaler(), Normalizer(), MinMaxScaler(), RobustScaler()
if scaler_select:
    X_train, X_test = scaleSelector(x_train, x_test, StandardScaler())
else:
    X_train, X_test = x_train, x_test
Y_train, X_valid, Y_valid = y_train, None, None

if kfold <= 1:
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, random_state = 5)

# Model Function:
#      Linear: LinearRegression(), LogisticRegression(), Perceptron(), Ridge(), Lasso()
#      Ensemble: RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()
#      Tree: DecisionTreeClassifier()
#      Neighbors: KNeighborsClassifier()
#      SVM: SVC()

# Random Seed
np.random.seed()

for modelSelection, numModels in modelCreation.items():
    totalModels += numModels
    print(modelSelection)
    for _ in tqdm(range(0,numModels)):

        if modelSelection == 'lightgbm':
            params, model = getModelLightGBM(previousModel, params)

        elif modelSelection == 'xgboost':
            params, model = getModelXGBoost(previousModel, params)

        elif modelSelection == 'lasso':
            # Lasso Model Generator
            params, model = getModelLasso(previousModel, params)

        elif modelSelection == 'ridge':
            # Ridge Model Generator
            params, model = getModelLasso(previousModel, params)

        elif modelSelection == 'knn':
            # KNN Model Generator
            params, model = getModelKNN(previousModel, params)

        elif modelSelection == 'gradboost':
            # Gradient Boosting Generator
            params, model = getModelGradientBoosting(previousModel, params)

        elif modelSelection == 'svc':
            # SVC Model Generator
            params, model = getModelSVC(previousModel, params)

        elif modelSelection == 'adaboost':
            # AdaBoost with DecisionTreeClassifier Model Generator
            params, model = getModelAdaBoostTree(previousModel, params)

        elif modelSelection == 'decisiontree':
            # DecisionTreeClassifier Model Generator
            params, model = getModelDecisionTree(previousModel, params)

        elif modelSelection == 'randomforest':
            # RandomForest Model Generator
            params, model = getModelRandomForest(previousModel, params)


        # Model Generation based off paramList and modelList
        Pred_train, Pred_valid, Pred_test, kfold_DF = modelSelector(X_train, Y_train, X_valid, Y_valid,
                                                                    X_test, train_full, model, kfold = kfold,
                                                                    kfold_type = kfold_type,
                                                                    modeltype = modelSelection)
        if not train_full:
            analysisDF = dataFrameUpdate(params, Y_train, Y_valid, Pred_train, Pred_valid, analysisDF, kfold_DF,
                                        kfold_update)
        if ensemble: ensembleList.append(Pred_test)

if not train_full:
    if kfold > 1:
        train_auc = analysisDF['Train Auc'].apply(lambda x: x.split('-')[0].strip()).astype('float64')
        valid_auc = analysisDF['Valid Auc'].apply(lambda x: x.split('-')[0].strip()).astype('float64')
    else:
        train_auc = analysisDF['Train Auc']
        valid_auc = analysisDF['Valid Auc']

    print(train_auc.max(), valid_auc.max())
    plt.plot(range(0, totalModels), train_auc, 'b', label = 'Train Auc')
    plt.plot(range(0, totalModels), valid_auc, 'r', label = 'Valid Auc')
    plt.show()
if not previousModel:
    analysisDF = analysisDF.sort_values(['Valid Auc','Train Auc'], ascending = False)
    analysisDF.to_csv('//Users/jeromydiaz/Desktop/Titanic_AnalysisDF.csv')

# Writes final submission file
if train_full: trainFullSubmission(Pred_test, test['PassengerId'], ensemble, ensembleList)


analysisDF.sort_values(['Valid Auc','Train Auc'], ascending = False).head(20)
