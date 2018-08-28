import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import backend as K
from keras import optimizers


def aucFunction(y_given, pred_given):
    '''
    Function aucFunction: returns the area under the ROC curve.  Useful metric for model tuning
        Input: 
            y_given: ground truth y values
            pred_given: predicted y values
        Return: 
            roc_auc: float - area under ROC curve. 

    '''
    falsePositive, truePositive, threshold = roc_curve(y_given, pred_given)
    roc_auc = auc(falsePositive, truePositive)
    
    return roc_auc


def seriesMeanSTD(df):
    '''
    Function seriesMeanSTD: returns the mean and standard deviation of a pandas series in a combined string
        input: 
            x: x - 
        return: x -  
    '''
    index_list = df.count().keys()
    df_mean = df.mean().get_values()
    df_std = df.std().get_values()
    comb_mean_std = []

    for v1, v2 in zip(df_mean, df_std):
        comb_mean_std.append(("%.6f" % v1) + ' - ' + ("%.6f" % v2))

    return pd.Series(comb_mean_std, index_list)


def getRandomNumber(firstNumber, lastNumber, getType = 'int'):
    '''
    Function getRandomNumber: Randomly generates one number depending on arguments
        Input: 
            first_number: int/float - Lower limit for range of random values needed
            last_number: int/float - Upper limit for range of random values needed
            get_type: string - The type of number you want returned
        Return: 
            int/float - random value based off arguments

        getType options:
            'int' returns a random int between arg1 and arg2
            'float' returns a random float between arg1 and arg2 (Do not use for < 0.1)
            'exp' returns a random float with exponential power.  All numbers will be 1^n.  n = range(arg1,arg2)
            'exp_random' returns a value similar to 'exp', however the 1 in (1^n) is also randomized between 1-9

    '''
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

    
def getRandomFromList(input_list):
    '''
    Function getRandomFromList: Randomly selects a single value from a list
        Input: 
            input_list: list - array-like list of strings, ints or floats
        Return: 
            string/int/float - a random single string, int or float depending on list type

    '''
    return input_list[getRandomNumber(0, len(input_list) - 1)]


def getSavedParams(load_index):
    '''
    Function getSavedParams: Obtains previously saved parameters and model_type to be use for training
        Input: 
            load_index: int - the index from a dataframe of models already collected
        Return:
            params: dict - recovers the parameters from the saved dataframe using load_index
            model_selection: string - the model_type ussd to determine which model to use the parameters in
            model_selection_mod: string - addition information for model_type if required

    '''
    # Loads data from a previously saved dataframe (in csv format)
    analysisDF = pd.read_csv('//Users/jeromydiaz/Desktop/Titanic_AnalysisDF.csv')
    
    # Load a single model (row) into a series
    s = analysisDF[analysisDF['Unnamed: 0'] == load_index].dropna(axis = 1).T.squeeze()
    
    # Initialize list of features to be dropped from series
    dropList = ['Unnamed: 0','Valid Accuracy', 'Valid Auc', 'Valid Loss', 'Train Accuracy',
                           'Train Auc', 'Train Loss', 'verbose', 'Model_type']
    
    # Splits model type into components 
    modelSplit = s['Model_type'].split('_')
    
    if len(modelSplit) > 1:
        modelSelection = modelSplit[0]
        modelSelectionMod = modelSplit[1]
    else:
        modelSelection = modelSplit[0]
        modelSelectionMod = None
    
    # Drops features from drop_list
    for columnName in dropList:
        if columnName in s.keys():
            s = s.drop(columnName)
            
    params = s.to_dict()
    
    # Converts all features needed to ints and floats to insure CSV data is in correct format
    toIntList = ['max_bin', 'max_depth', 'min_data', 'num_leaves', 'num_trees',
                'bagging_freq', 'n_estimators', 'degree', 'max_features', 'n_neighbors', 'p', 'min_child_weight',
                'max_leaf_nodes', 'min_child_weight', 'seed', 'min_child_samples', 'subsample_freq', 'batch_size',
                'n_layers', 'n_layers_neurons_1', 'n_layers_neurons_2', 'n_layers_neurons_3', 'n_layers_neurons_4',
                'epoch']
    
    for feature in toIntList:
        if feature in params:
            params[feature] = int(params[feature])
    
    toFloatList = ['gamma']
    for feature in toFloatList:
        if feature in params:
            params[feature] = float(params[feature])    
            
    return params, modelSelection, modelSelectionMod


def trainFullSubmission(pred_test, ID, ensemble, ensembleList):
    '''
    Function trainFullSubmission: trains and outputs a submission results from model
        Input: 
            pred_test: int/boolean - prediction of y values 
            ID: string - the feature name for y values for submission
            ensemble: boolean - if true, will ensemble all models and take the average prediction for each row
            ensembleList: list - array-like list of predictions for each model in ensemble
        Return:
            void: However, a file will be made for submission at desginated path.

    '''
    # Ensemble Models Together
    if ensemble: 
        pred_test = np.where(np.array(ensembleList).mean(axis = 0) > 0.5, 1,0)
        print("Ensemble Predition: " + str(np.unique(pred_test, return_counts = True)))

    # Final Submission - Change to ID
    finaldf = pd.DataFrame(ID)
    finaldf['Survived'] = pred_test
    finaldf.set_index('PassengerId', inplace = True)
    finaldf.to_csv('/Users/jeromydiaz/Desktop/TitanicFinalSub.csv')


def scaleSelector(x_train, x_test, scaler):
    '''
    Function scaleSelector: transforms train/test data using the scaler options below
        Input: 
            x_train: dataFrame - train data
            x_test: dataFrame - test data
            scaler: object - scaler object used to transform the data
        Return:
            x_train: dataFrame - returns transformed dataFrame
            x_test: dataFrame - returns transformed dataFrame

        Scaler Options:
            Normalizer() - normalizes the data across [rows]
            StandardScaler() - standardizes your data with a mean of 0 and a standard deviation of 1 across [columns]
            MinMaxScaler() - alternative to standard which transforms data into bins instead of individual points [columns]
            RobustScaler() - transforms data using a quartile based scaling to decrease the impact of outliers [columns]

    '''
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test


def dataFrameUpdate(params, y_train, y_valid, pred_train, pred_valid, analysis_df, kfold_df, update_Kfold = False):
    '''
    Function dataFrameUpdate: updates specified dataFrame with parameters / metric scores from the model generated
        Input: 
            params: dict - parameters of model 
            y_train: dataFrame - training - ground truth values
            y_valid: dataFrame - validation set - ground truth values
            pred_train: list - training - predicted values
            pred_valid: list - validation set - predicted values
            analysis_df: dataFrame - to be updated
                if update_kfold == True: it will use the kfold_df to update the score using the mean and std of the models
                if update_kfold == False: it will update the metric scores with no averaging (kfold = 1)
            kfold_df: dataFrame - seperate dataFrame used to store the results of each fold to be averaged/std
            update_Kfold: boolean - updates main dataFrame with kfold data if true

        Return: 
            updateDF: dataFrame - returns either main dataFrame or kfold dataFrame updating the data for each fold

    '''
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

############### NN Model Selector ####################
def getOptimizer(optimizer, learning_rate):
    """
    Current supported opimtizers are 'adam', 'rmsprop', 'sgd', 'adagrad'.  These are fed into the model compile in 
    NNModelSelector function. 
    
    """
    
    if optimizer == 'adam':
        return optimizers.Adam(lr = learning_rate)
    elif optimizer == 'rmsprop':
        return optimizers.RMSprop(lr = learning_rate)
    elif optimizer == 'sgd':
        return optimizers.SGD(lr = learning_rate)
    elif optimizer == 'adagrad':
        return optimizers.Adagrad(lr = learning_rate)

def NNModelSelector(params, input_shape):
    """
    Creates the computational graph for the neural network.  The function returns the model to be used in the 
    modelSelector function (which all models are passed through).
    
    """
    
    K.clear_session()
    in_x = Input(shape = (input_shape,))
    X = Dense(params['n_layers_neurons_1'], kernel_initializer = params['kernel_initializer'], 
              activation = 'relu')(in_x)
    
    # Makes 'n_layers' Fully Connected layers 
    for i in range(2,params['n_layers'] + 1):             
        X = Dense(params['n_layers_neurons_' + str(i)], kernel_initializer = params['kernel_initializer'], 
                  activation = 'relu')(X)

    Y = Dense(1, kernel_initializer = params['kernel_initializer'], activation = 'sigmoid')(X)
    model = Model(inputs = in_x, outputs = Y)

    optimizer = getOptimizer(params['optimizer'], params['learning_rate'])
    model.compile(optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model  

############### Standard Model Selector ####################
def modelSelector(x_train, y_train, x_valid, y_valid, x_test, params, train_full, Model, kfold = 1, 
                  kfold_type = 'normal', modeltype = None):
    '''
    Function modelSelector: fits models and predicts outputs
        Input: 
            x_train: dataFrame - training - data
            y_train: dataFrame - training - ground truth values
            x_valid: dataFrame - validation set - data
            y_valid: dataFrame - validation set - ground truth values
            x_test: dataFrame - data to be predicted using model
            params: dict - parameters of model (if required)
            train_full: boolean - determines if x_test should be evaluated
            Model: object - premade model object ready for fitting (set to None, if not required)
            kfold: int - # of folds for k-fold cross validation (if equals 1, pre-split before this function)
            kfold_type: tuple - 
                kfold_type[0]: string - kfold type 'normal','normal_repeat','strat','strat_repeat'
                kfold_type[1]: int - if 'normal_repeat' or 'strat_repeat', then [1] will be the number of repeats
            modeltype: string - type of model to be created 
        Return: 
            pred_train: list - prediction of training set
            pred_valid: list - prediction of validation set
            pred_test: list - prediction of test set (if train_full == false, set to None)
            kfold_mean_std_S: Series - mean and standard deviation of all folds per model

    '''
    # Determines if using KFold or single split
    if kfold > 1: 
        kfold_DF = pd.DataFrame()
        kfold_pred_train = []
        kfold_pred_valid = []
        kfold_params = {}
        # kfold_type[0] = which kfold function to use
        # kfold_type[1] = # of repeats (if applicable)
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
                
        elif modeltype == 'neuralnetwork':
            model = NNModelSelector(params, X_train.shape[1])
            model.fit(X_train, Y_train, validation_data = (X_valid, Y_valid), epochs = params['epoch'],
                      batch_size = params['batch_size'], verbose = 0)

            pred_train = model.predict(X_train, batch_size = params['batch_size'])
            pred_valid = model.predict(X_valid, batch_size = params['batch_size'])
            if train_full: pred_test = model.predict(x_test, batch_size = params['batch_size'])
 
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
