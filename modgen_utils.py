import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_curve, auc, r2_score
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
    false_positive, true_positive, threshold = roc_curve(y_given, pred_given)
    roc_auc = auc(false_positive, true_positive)

    return roc_auc


def seriesMeanSTD(df):
    '''
    Function seriesMeanSTD: returns the mean and standard deviation of a pandas series in a combined string
        input:
            x: x -
        return: x -
    '''
    index_list = df.count().keys()
    mean_of_DF = df.mean().get_values()
    std_of_DF = df.std().get_values()
    concat_mean_std = []

    for v1, v2 in zip(mean_of_DF, std_of_DF):
        concat_mean_std.append(("%.6f" % v1) + ' _ ' + ("%.6f" % v2))

    return pd.Series(concat_mean_std, index_list)


def getRandomNumber(lower_range_limit, upper_range_limit, random_type = 'int'):
    '''
    Function getRandomNumber: Randomly generates one number depending on arguments
        Input:
            lower_range_limit: int/float - Lower limit for range of random values needed
            upper_range_limit: int/float - Upper limit for range of random values needed
            random_type: string - The type of number you want returned
        Return:
            int/float - random value based off arguments

        random_type options:
            'int' returns a random int between arg1 and arg2
            'float' returns a random float between arg1 and arg2 (Do not use for < 0.1)
            'exp' returns a random float with exponential power.  All numbers will be 1^n.  n = range(arg1,arg2)
            'exp_random' returns a value similar to 'exp', however the 1 in (1^n) is also randomized between 1-9

    '''
    # Returns int between first and last number
    if random_type == 'exp' or random_type == 'exp_random':
        if lower_range_limit > 0 and upper_range_limit > 0:
            exponent_range = np.logspace(lower_range_limit, upper_range_limit, (abs(upper_range_limit) - abs(lower_range_limit) + 1))
        elif lower_range_limit < 0 and upper_range_limit < 0:
            exponent_range = np.logspace(lower_range_limit, upper_range_limit, (abs(lower_range_limit) - abs(upper_range_limit) + 1))
        elif lower_range_limit == 0 or upper_range_limit == 0:
            print('Cannot use 0 as an argument - Use 1 instead')
        else:
            exponent_range = np.logspace(lower_range_limit, upper_range_limit, (abs(lower_range_limit) + abs(upper_range_limit) + 1))
        if random_type == 'exp_random': exponent_range = exponent_range * np.random.randint(1, 9 + 1)
        return exponent_range[np.random.randint(0, len(exponent_range))]
    # Returns float between first and last number
    elif random_type == 'float':
        return (upper_range_limit - lower_range_limit) * np.random.rand() + lower_range_limit
    else:
        return np.random.randint(lower_range_limit, upper_range_limit + 1)


def getRandomFromList(input_list):
    '''
    Function getRandomFromList: Randomly selects a single value from a list
        Input:
            input_list: list - array-like list of strings, ints or floats
        Return:
            string/int/float - a random single string, int or float depending on list type

    '''
    return input_list[getRandomNumber(0, len(input_list) - 1)]


def getSavedParams(path,load_index):
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
    analysis_DF = pd.read_csv(path + 'analysis_df.csv')

    # Load a single model (row) into a series
    s = analysis_DF[analysis_DF['Unnamed: 0'] == load_index].dropna(axis = 1).T.squeeze()

    # Initialize list of features to be dropped from series
    features_to_drop = ['Unnamed: 0','Valid Accuracy', 'Valid Auc', 'Valid Loss', 'Train Accuracy',
                           'Train Auc', 'Train Loss', 'verbose', 'Model_type']

    # Splits model type into components
    model_type_split = s['Model_type'].split('_')

    if len(model_type_split) > 1:
        model_selector = model_type_split[0]
        model_selector_mod = model_type_split[1]
    else:
        model_selector = model_type_split[0]
        model_selector_mod = None

    # Drops features from drop_list
    for feature_name in features_to_drop:
        if feature_name in s.keys():
            s = s.drop(feature_name)

    params = s.to_dict()

    # Converts all features needed to ints and floats to insure CSV data is in correct format
    convert_to_int = ['max_bin', 'max_depth', 'min_data', 'num_leaves', 'num_trees',
                'bagging_freq', 'n_estimators', 'degree', 'max_features', 'n_neighbors', 'p', 'min_child_weight',
                'max_leaf_nodes', 'min_child_weight', 'seed', 'min_child_samples', 'subsample_freq', 'batch_size',
                'n_layers', 'n_layers_neurons_1', 'n_layers_neurons_2', 'n_layers_neurons_3', 'n_layers_neurons_4',
                'epoch']

    for feature_name in convert_to_int:
        if feature_name in params:
            params[feature_name] = int(params[feature_name])

    convert_to_float = ['gamma']
    for feature_name in convert_to_float:
        if feature_name in params:
            params[feature_name] = float(params[feature_name])

    return params, model_selector, model_selector_mod


def trainFullSubmission(pred_test, submission_name_series, submission_column_names, ensemble, ensemble_predictions, path):
    '''
    Function trainFullSubmission: trains and outputs a submission results from model
        Input:
            pred_test: int/boolean - prediction of y values
            ID: string - the feature_name name for y values for submission
            ensemble: boolean - if true, will ensemble all models and take the average prediction for each row
            ensemble_predictions: list - array-like list of predictions for each model in ensemble
        Return:
            void: However, a file will be made for submission at desginated path.

    '''
    # Ensemble Models Together
    if ensemble:
        pred_test = np.where(np.array(ensemble_predictions).mean(axis = 0) > 0.5, 1,0)
        print("Ensemble Predition: " + str(np.unique(pred_test, return_counts = True)))

    # Final Submission - Change to ID
    test_submission_DF = pd.DataFrame(submission_name_series)
    test_submission_DF[submission_column_names[1]] = pred_test
    test_submission_DF.set_index(submission_column_names[0], inplace = True)
    test_submission_DF.to_csv(path + 'prediction_submission.csv')


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


def dataFrameUpdate(is_classifier, params, y_train, y_valid, pred_train, pred_valid, analysis_df,
                    kfold_DF, update_kfold = False):
    '''
    Function dataFrameUpdate: updates specified dataFrame with parameters / metric scores from the model generated
        Input:
            params: dict - parameters of model
            y_train: dataFrame - training - ground truth values
            y_valid: dataFrame - validation set - ground truth values
            pred_train: list - training - predicted values
            pred_valid: list - validation set - predicted values
            analysis_df: dataFrame - to be updated
                if update_kfold == True: it will use the kfold_DF to update the score using the mean and std of the models
                if update_kfold == False: it will update the metric scores with no averaging (kfold_number_folds = 1)
            kfold_DF: dataFrame - seperate dataFrame used to store the results of each fold to be averaged/std
            update_kfold: boolean - updates main dataFrame with kfold_number_folds data if true

        Return:
            update_DF: dataFrame - returns either main dataFrame or kfold_number_folds dataFrame updating the data for each fold

    '''
    update_DF = analysis_df
    update_to_DF = params
    if pred_train is not None:
        if (np.isnan(pred_train).any() or np.isnan(pred_valid).any()):
            s = pd.Series(update_to_DF)
            update_DF = update_DF.append(s, ignore_index=True)
            return update_DF

    if update_kfold:
        if is_classifier:
            update_to_DF['Train Accuracy'] = kfold_DF['Train Accuracy']
            update_to_DF['Valid Accuracy'] = kfold_DF['Valid Accuracy']
        update_to_DF['Train Loss'] = kfold_DF['Train Loss']
        update_to_DF['Valid Loss'] = kfold_DF['Valid Loss']
        update_to_DF['Train Auc(C)-R2(R)'] = kfold_DF['Train Auc(C)-R2(R)']
        update_to_DF['Valid Auc(C)-R2(R)'] = kfold_DF['Valid Auc(C)-R2(R)']

    else:
        if is_classifier:
            update_to_DF['Train Accuracy'] = accuracy_score(y_train, pred_train)
            update_to_DF['Valid Accuracy'] = accuracy_score(y_valid, pred_valid)
            update_to_DF['Train Auc(C)-R2(R)'] = aucFunction(y_train, pred_train)
            update_to_DF['Valid Auc(C)-R2(R)'] = aucFunction(y_valid, pred_valid)
        else:
            update_to_DF['Train Auc(C)-R2(R)'] = r2_score(y_train, pred_train)
            update_to_DF['Valid Auc(C)-R2(R)'] = r2_score(y_valid, pred_valid)
        update_to_DF['Train Loss'] = mean_absolute_error(y_train, pred_train)
        update_to_DF['Valid Loss'] = mean_absolute_error(y_valid, pred_valid)

    s = pd.Series(update_to_DF)
    update_DF = update_DF.append(s, ignore_index=True)

    return update_DF

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

    Y = Dense(1, kernel_initializer = params['kernel_initializer'], activation = params['final_activation'])(X)
    model = Model(inputs = in_x, outputs = Y)

    optimizer = getOptimizer(params['optimizer'], params['learning_rate'])
    model.compile(optimizer, loss = params['loss_optimize'], metrics = ['accuracy'])

    return model

############### Standard Model Selector ####################
def modelSelector(x_train, y_train, x_valid, y_valid, x_test, params, train_test_submission, Model, is_classifier, kfold_number_of_folds = 1, kfold_type = ('normal',1), modeltype = None):
    '''
    Function modelSelector: fits models and predicts outputs
        Input:
            x_train: dataFrame - training - data
            y_train: dataFrame - training - ground truth values
            x_valid: dataFrame - validation set - data
            y_valid: dataFrame - validation set - ground truth values
            x_test: dataFrame - data to be predicted using model
            params: dict - parameters of model (if required)
            train_test_submission: boolean - determines if x_test should be evaluated
            Model: object - premade model object ready for fitting (set to None, if not required)
            kfold_number_folds: int - # of folds for k-fold cross validation (if equals 1, pre-split before this function)
            kfold_type: tuple -
                kfold_type[0]: string - kfold_number_folds type 'normal','normal_repeat','strat','strat_repeat'
                kfold_type[1]: int - if 'normal_repeat' or 'strat_repeat', then [1] will be the number of repeats
            modeltype: string - type of model to be created
        Return:
            pred_train: list - prediction of training set
            pred_valid: list - prediction of validation set
            pred_test: list - prediction of test set (if train_test_submission == false, set to None)
            kfold_concat_mean_std: Series - mean and standard deviation of all folds per model

    '''
    # Determines if using kfold_number_folds or single split
    if kfold_number_of_folds > 1:
        kfold_DF = pd.DataFrame()
        kfold_pred_train = []
        kfold_pred_valid = []
        kfold_params = {}
        # kfold_type[0] = which kfold_number_folds function to use
        # kfold_type[1] = # of repeats (if applicable)
        if kfold_type[0] == 'normal': kfold_selector = KFold(n_splits = kfold_number_of_folds, random_state = 0)
        if kfold_type[0] == 'normal_repeat': kfold_selector = RepeatedKFold(n_splits = kfold_number_of_folds, n_repeats = kfold_type[1],
                                                                       random_state = 0)
        if kfold_type[0] == 'strat': kfold_selector = StratifiedKFold(n_splits = kfold_number_of_folds, random_state = 0)
        if kfold_type[0] == 'strat_repeat': kfold_selector = RepeatedStratifiedKFold(n_splits = kfold_number_of_folds,
                                                                                n_repeats = kfold_type[1],
                                                                                random_state = 0)
        iteration_loop = kfold_selector.split(x_train, y_train)
    else:
        # iteration_loop is a throwaway to pass into for single split
        X_train, X_valid, Y_train, Y_valid = x_train, x_valid, y_train, y_valid
        iteration_loop = zip([None],[None])

    # Loop for kfold_number_folds or single split
    for train_index, valid_index in iteration_loop:
        if kfold_number_of_folds > 1:
            X_train, X_valid = x_train[train_index], x_train[valid_index]
            Y_train, Y_valid = y_train[train_index], y_train[valid_index]

        if modeltype == 'lightgbm':
            # If you keep.. remember to change x_train to X_train (CAPS)
            model = Model
            model.fit(X_train, Y_train, eval_set = (X_valid, Y_valid), eval_metric = params['metric'],
                      early_stopping_rounds = 40, verbose = False)
            pred_train = model.predict(X_train, num_iteration = model.best_iteration_)
            pred_valid = model.predict(X_valid, num_iteration = model.best_iteration_)
            if train_test_submission: pred_test = model.predict(x_test, num_iteration = model.best_iteration_)

        elif modeltype == 'neuralnetwork':
            model = NNModelSelector(params, X_train.shape[1])
            model.fit(X_train, Y_train, validation_data = (X_valid, Y_valid), epochs = params['epoch'],
                      batch_size = params['batch_size'], verbose = 0)

            pred_train = model.predict(X_train, batch_size = params['batch_size'])
            pred_valid = model.predict(X_valid, batch_size = params['batch_size'])
            if train_test_submission: pred_test = model.predict(x_test, batch_size = params['batch_size'])

        else:
            model = Model
            model.fit(X_train, Y_train)
            pred_train = model.predict(X_train)
            pred_valid = model.predict(X_valid)
            if train_test_submission: pred_test = model.predict(x_test)
        if is_classifier: pred_train = np.where(pred_train > 0.5, 1, 0)
        if is_classifier: pred_valid = np.where(pred_valid > 0.5, 1, 0)

        if kfold_number_of_folds > 1:
            kfold_DF = dataFrameUpdate(is_classifier, kfold_params, Y_train, Y_valid, pred_train, pred_valid, kfold_DF,
                                       None)
            kfold_pred_train.append(pred_train)
            kfold_pred_valid.append(pred_valid)

    if train_test_submission:
        if is_classifier: pred_test = np.where(pred_test > 0.5, 1, 0)
    else:
        pred_test = None

    if kfold_number_of_folds > 1:
        # pred_train = np.where(np.array(kfold_pred_train).mean(axis = 0) > 0.5, 1,0)
        # pred_valid = np.where(np.array(kfold_pred_valid).mean(axis = 0) > 0.5, 1,0)
        pred_train = None
        pred_valid = None
        kfold_mean_DF = kfold_DF.mean()
        kfold_concat_mean_std = seriesMeanSTD(kfold_DF)
    else:
        kfold_mean_DF = None
        kfold_concat_mean_std = None
    return pred_train, pred_valid, pred_test, kfold_concat_mean_std
