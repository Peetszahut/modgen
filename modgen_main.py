from modgen_utils import *
from modgen_classifier_models import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

##################### feature_name Engineering Code Goes Here #####################

train = pd.read_csv("~/train.csv")
test = pd.read_csv("~/test.csv")

##################### feature_name Engineering Code Goes Here #####################

######## Model Options ########
### K-Fold Options
# 'normal', 'strat', 'normal_repeat', 'strat_repeat' - (type, # repeats)
use_kfold_CV = True
kfold_number_of_folds = 4
kfold_distribution = 'normal'
kfold_repeats = 1

### Scaler Option: StandardScaler(), Normalizer(), MinMaxScaler(), RobustScaler()
# If scaler_select = None, then no scaling will be done
scaler_select = StandardScaler()

### Model creation options
# If use_previous_model = False: Use new models_to_be_created dictionary to make models, else, use previous index to get parameters
# If train_test_submission = True: Train data on test set and make submission file of results
# If ensemble = True: Ensemble all previous index models together
use_previous_model = False
train_test_submission = False
ensemble = False

# Initialization of Lists and DFs
ensemble_predictions = []
analysis_DF = pd.DataFrame()
kfold_DF = pd.DataFrame()
params = {}
total_models = 0

if use_previous_model:
    params, model_selector, model_selector_mod = getSavedParams(load_index = 150)
    models_to_be_created = {model_selector : 1}
else:
    models_to_be_created = {
                        'neuralnetwork': 5,
                        'lightgbm'     : 2000,
                        'xgboost'      : 1500,
                        'lasso'        : 200,
                        'ridge'        : 200,
                        'knn'          : 200,
                        'gradboost'    : 500,
                        'svc'          : 250,
                        'decisiontree' : 500,
                        'randomforest' : 200
                    }

### K-Fold Cross Validation Inputs
# If a prevous_model is being loaded, then it automatically turns off kfold_CV
if use_previous_model: use_kfold_CV = False
if use_kfold_CV:
    kfold_type = (kfold_distribution, kfold_repeats)
else:
    kfold_number_of_folds = 1
    kfold_type = None

### Scaler Function Inputs
if scaler_select:
    X_train, X_test = scaleSelector(x_train, x_test, scaler_select)
else:
    X_train, X_test = x_train, x_test
Y_train, X_valid, Y_valid = y_train, None, None

# If kfold_number_of_folds == 1: Split the data using train_test_split
if kfold_number_of_folds <= 1:
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, random_state = 5)

# Model Function:
#      Linear: LinearRegression(), LogisticRegression(), Perceptron(), Ridge(), Lasso()
#      Ensemble: RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()
#      Tree: DecisionTreeClassifier()
#      Neighbors: KNeighborsClassifier()
#      SVM: SVC()

# Random Seed
np.random.seed()

for model_selector, number_of_models in models_to_be_created.items():
    total_models += number_of_models
    print(model_selector)
    for _ in tqdm(range(0,number_of_models)):

        if model_selector == 'lightgbm':
            params, model = getModelLightGBM(use_previous_model, params)

        elif model_selector == 'xgboost':
            params, model = getModelXGBoost(use_previous_model, params)

        elif model_selector == 'neuralnetwork':
            params, model = getModelNeuralNetwork(use_previous_model, params)

        elif model_selector == 'lasso':
            # Lasso Model Generator
            params, model = getModelLasso(use_previous_model, params)

        elif model_selector == 'ridge':
            # Ridge Model Generator
            params, model = getModelLasso(use_previous_model, params)

        elif model_selector == 'knn':
            # KNN Model Generator
            params, model = getModelKNN(use_previous_model, params)

        elif model_selector == 'gradboost':
            # Gradient Boosting Generator
            params, model = getModelGradientBoosting(use_previous_model, params)

        elif model_selector == 'svc':
            # SVC Model Generator
            params, model = getModelSVC(use_previous_model, params)

        elif model_selector == 'adaboost':
            # AdaBoost with DecisionTreeClassifier Model Generator
            params, model = getModelAdaBoostTree(use_previous_model, params)

        elif model_selector == 'decisiontree':
            # DecisionTreeClassifier Model Generator
            params, model = getModelDecisionTree(use_previous_model, params)

        elif model_selector == 'randomforest':
            # RandomForest Model Generator
            params, model = getModelRandomForest(use_previous_model, params)


        # Model Generation based off paramList and modelList
        Pred_train, Pred_valid, Pred_test, kfold_DF = modelSelector(X_train, Y_train, X_valid, Y_valid,
                                                                    X_test, params, train_test_submission, model, kfold_number_of_folds = kfold_number_of_folds,
                                                                    kfold_type = kfold_type,
                                                                    modeltype = model_selector)

        analysis_DF = dataFrameUpdate(params, Y_train, Y_valid, Pred_train, Pred_valid, analysis_DF, kfold_DF,
                                    use_kfold_CV)
        if ensemble: ensemble_predictions.append(Pred_test)


if kfold_number_of_folds > 1:
    train_auc = analysis_DF['Train Auc'].apply(lambda x: x.split('-')[0].strip()).astype('float64')
    valid_auc = analysis_DF['Valid Auc'].apply(lambda x: x.split('-')[0].strip()).astype('float64')
else:
    train_auc = analysis_DF['Train Auc']
    valid_auc = analysis_DF['Valid Auc']

print(train_auc.max(), valid_auc.max())
plt.plot(range(0, total_models), train_auc, 'b', label = 'Train Auc')
plt.plot(range(0, total_models), valid_auc, 'r', label = 'Valid Auc')
plt.show()
if not use_previous_model:
    analysis_DF = analysis_DF.sort_values(['Valid Auc','Train Auc'], ascending = False)
    analysis_DF.to_csv('//Users/jeromydiaz/Desktop/Titanic_AnalysisDF.csv')

# Writes final submission file
if train_test_submission: trainFullSubmission(Pred_test, test['PassengerId'], ensemble, ensemble_predictions)

analysis_DF.sort_values(['Valid Auc','Train Auc'], ascending = False)
