"""
This file holds all of the model not related to neural networks.  New models can be added with relative ease by 
making a new function and passing in previousModel and params.  Params is a dictionary which can store any parameters
to be fed into the model. Currently, only classifiers are included, but regressors could easily be added in time.

Variables:
    params = Dictionary with parameters to set into models
    previousModel = Boolean set in the main program. This boolean lets the program know if you are grabbing from 
    a previous index or if you are making new models.
    
Classifier Models:
    Logistic Regression
    Ridge
    Lasso
    Decision Tree
    KNN
    SVC

Ensembles:
    Random Forest
    
Boosting Models:
    AdaBoost
    GradientBoost
    LightGBM
    XGBoost

"""
import pandas as pd
import numpy as np
from modgen_utils import *
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def getModelRidge(previousModel, params):
    """
    Ridge model used as L2 Regularization.  The model itself will keep the coef from blowing up or shrinking
    down to a superlow or superhigh number causing massive overfitting of the data.  Generally alpha will be
    between 1e-15 and 1e-3.  Can use all features at once, but computationally heavy as you add more features.
    
    """
    if not previousModel:
        params = {}
        params['Model_type'] = 'ridge'
        params['alpha'] = getRandomNumber(-15,2, getType = 'exp')

    model = Ridge(alpha = params['alpha'], max_iter = 1e6, random_state = 0)
    
    return params, model


def getModelLasso(previousModel, params):
    """
    Lasso model used as L1 regularization.  The model will create more and more sparse coef by deciding which 
    features are important and ignorning the rest.  The term lasso means lassoing up all the good 
    features and leaving out the bad ones.
    
    """
    if not previousModel:
        params = {}
        params['Model_type'] = 'lasso'
        params['alpha'] = getRandomNumber(-15,2, getType = 'exp')

    model = Lasso(alpha = params['alpha'], max_iter = 1e6, random_state = 0)  
    
    return params, model


def getModelKNN(previousModel, params):
    """
    K Nearest Neighbor Model used as an unsupervised classifier.  Generating decision boundaries based off the 
    probability of each points to its nearest neighbors to determine class boundaries.
    
    """
    if not previousModel:
        params = {}
        params['Model_type'] = 'knn'
        params['n_neighbors'] = getRandomNumber(1,30)
        params['p'] = getRandomNumber(1,2)

    model = KNeighborsClassifier(n_neighbors = params['n_neighbors'], p = params['p'])
    
    return params, model


def getModelGradientBoosting(previousModel, params):
    """
    Gradient Boosting Model using weaker regression trees to distribute a high variance across an ensemble of 
    classifiers.  Used as the premise for many gradient boosting based algorithms.
    
    """
    if not previousModel:
        params = {}
        params['Model_type'] = 'gradboost'
        params['learning_rate'] = getRandomNumber(-3,2, getType = 'exp_random')
        params['n_estimators'] = getRandomNumber(1,500)
        params['max_depth'] = getRandomNumber(1,32)
        params['min_samples_split'] = getRandomNumber(0.1,1, getType = 'float')
        params['min_samples_leaf'] = getRandomNumber(0.1,0.5, getType = 'float')

    model = GradientBoostingClassifier(learning_rate = params['learning_rate'], 
                                       n_estimators = params['n_estimators'],
                                       max_depth = params['max_depth'], 
                                       min_samples_split = params['min_samples_split'],
                                       min_samples_leaf = params['min_samples_leaf'],
                                       random_state = 0)
    return params, model


def getModelSVC(previousModel,params):
    """
    Support Vector Machine model is used for linear / kernel modeling.  Each parameter is taken care of for each of
    the options for each kernel. Kernel trick (rbf) used to move the data to a higher dimensionality to seperate 
    non-linearly seperable data.

    """
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
            params['degree'] = getRandomNumber(1,4) # Only used on POLY - [1,2,3,4]
        else:
            params['degree'] = 3


    model = SVC(kernel = params['kernel'], gamma = params['gamma'], C = params['C'], degree = params['degree'],
               max_iter = 1e6, random_state = 0)
    
    return params, model


def getModelAdaBoostTree(previousModel, params):
    """
    Ada Boosting with Decision Tree Classifier.  Uses adaboost technique to boost weak classifiers from decision trees.
    Has shown good results in the past models.
    
    """
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
    """
    Decision Tree Classifiers.  Very fast classifier computationally.  Results have been promising (in top 90%) 
    
    """
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
    """
    Random Forest models is an ensemble of weaker decision tree classifiers (with large max_depth) 
    to build one strong classifier n_estimators determines how many trees will be built. 
    
    """
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
    """
    XGBoost model boosting technique. Best results for this program have been with XGBoost.  This is the sklearn
    implementation as opposed to the original implementation (with different parameter names).
    
    """
    if not previousModel:
        params = {}
        params['Model_type'] = 'xgboost'
        params['learning_rate'] = getRandomNumber(-5,1, getType = 'exp_random')
        params['min_child_weight'] = getRandomNumber(1,10)
        params['max_depth'] = getRandomNumber(1,10)
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
    """
    Microsoft's LightGBM model with sklearn implementation. Can do 3 different boosting types at random.  If 
    you want to do only one boosting type, then take out the boosting types you dont want from the boosting type 
    list.  The available boosting types are 'dart' - dropout, 'gdbt' - gradient boosting, 'rf' - random forest.
    
    Early stopping is enabled, so n_estimators is set to 10k, but usually does not get to 10k iteration before early
    stopping takes effect.
    
    """
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


def getModelNeuralNetwork(previousModel, params):
    """
    Sets the paramters for how many layers to make and the amount of activation nodes.  Model is set to 'None' due to
    NNModelSelector creating the model with the designated parameters.
    
    """
    
    if not previousModel:
        params = {}
        params['Model_type'] = 'neuralnetwork'
        params['epoch'] = getRandomNumber(15, 50, 'int')
        params['kernel_initializer'] = getRandomFromList(['random_uniform', 'random_normal', 'glorot_normal',
                                                         'glorot_uniform'])
        params['optimizer'] = getRandomFromList(['adam', 'rmsprop', 'sgd', 'adagrad'])
        params['learning_rate'] = getRandomNumber(-3, 1, 'exp_random')
        params['batch_size'] = getRandomFromList([2,4,8,16,32,64])
        params['n_layers'] = getRandomNumber(1, 4, 'int')
        for i in range(1, params['n_layers'] + 1):
            params['n_layers_neurons_' + str(i)] = getRandomNumber(2, 12, 'int')
            
    # Model not instantiated like other models
    model = None        
    
    return params, model