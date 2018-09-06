# Modgen - Automatic Model Generator
## Overview of Modgen
This program was created for rapid feature engineering without the need to optimize each model.  Modgen is designed to develop a quick overview of how your updated features will react to each model.  You can use one specific algorithm or a wide variety (depending on your interests) with a random feature range which can be easily changed at anytime by the user.

<img src="docs/pictures/program_overview.png" height="250">

### Libraries Used
* Pandas                    
* Numpy
* Matplotlib
* SkLearn
* Keras
* LightGBM
* XGBoost
* tqdm

### Algorithms Currently Implemented
* Neural Network
* LightGBM
* XGBoost
* Random Forest
* K Nearest Neighbor
* Support Vector Machine
* Decision Tree
* Grad Boost
* Lasso
* Ridge

## How It Works
### Data Upload and Feature Engineering
The main portion you will be coding on is shown below.  First, designate the path where your train/test set are stored (all other files will be saved to this same path). Play around all you want with your data in the section designated 'Feature Engineering Code Goes Here'.  When you are satisfied with the data you wish to feed into the model, assign the variables below (Do not split your data into train/validation set - will be done automatically in the code):
* x_train
* y_train
* x_test

```
######################### Path / Train / Test Data ###########################
path = '~/Documents/'
train = pd.read_csv(path + "train.csv", nrows = 1_000_000)
test = pd.read_csv(path + "test.csv")

##################### Feature Engineering Code Goes Here #####################



##################### Feature Engineering Code Goes Here #####################
```

## Model Options
The options below will be used to develop the models.  I will go through each one below to get the program ready.  If all options are set, then skip to section 'Model Maker'

### Classifier or Regression
Determines if the models will be Classifiers(Binary) or Regressors(Continuous).
```
# Classification or Regression
is_classifier = False
```
### Split / K-Fold Options
Determines if K-Fold Cross Validation will be used (if not used, then skip to next section for split).  If it is used, then below sets the number of folds, the distribution of the split and the number of repeated K-Folds if 'normal_repeat' or 'strat_repeat' is selected (strat = stratify).
```
### K-Fold Options:
# 'normal', 'strat', 'normal_repeat', 'strat_repeat' - (type, # repeats)
use_kfold_CV = False
kfold_number_of_folds = 4
kfold_distribution = 'normal'
kfold_repeats = 1
```

If use_kfold_CV is false, then the default will use the option below to split the data.  
```
### Data Split Options: Train/Validation split (Non-KFold)
# Percentage of total data to be used in the validation set (train set automatically set 1 - split_valid_size)
split_valid_size = 0.25
```

### Scale Selector Options
Selects a scaler option if required (copy/paste any of the 4 or type None).
```
### Scaler Option: StandardScaler(), Normalizer(), MinMaxScaler(), RobustScaler()
# If scaler_select = None, then no scaling will be done
scaler_select = StandardScaler()
```

### Model Creation Options
Options described below:
* use_previous_model = False: Use new models_to_be_created dictionary to make models to be developed, else, use previous index to get saved parameters / model.
* train_test_submission = True: Train data on test set and make submission file of results
* submission_column_names: The key and predicted value column names for submission file (only required if train_test_submission = True)
* ensemble = True: Ensemble all previous index models together [NOT CURRENTLY WORKING]
```
### Model Creation Options:
use_previous_model = False
train_test_submission = False
submission_column_names = ('key','fare_amount')
ensemble = False
```

### Model Creation
Main model creator
* use_previous_model = True: Insert an index number from previously saved analysis_df.csv.  Model automatically selected and remade for submission.
* use_previous_model = False: Creates models with random parameters for each model in dict.  The # is the amount of random models to create for each key. Can delete / comment out any model you do not want in your creator (Caution: neuralnetwork can take awhile).

```
if use_previous_model:
    params, model_selector, model_selector_mod = getSavedParams(path, load_index = 41)
    models_to_be_created = {model_selector : 1}
else:
    models_to_be_created = {
                        'lightgbm'     : 200,
                        'xgboost'      : 200,
                        'knn'          : 25,
                        'svm'          : 25,
                        'decisiontree' : 25,
                        'randomforest' : 25,
                        'neuralnetwork': 50,
                        'gradboost'    : 50,
                        # 'lasso'        : 500,
                        # 'ridge'        : 500,
                    }
```

## Model Maker
Run the rest of the program.  This process can take awhile depending on your computer (I usually run it in AWS with EC2/Jupyter).
Once complete, a graph will appear graphing AUC/R2 scores and giving the max train/validation score.  

<img src="docs/pictures/program_complete.png" height="450">

Run the code below to display the dataframe with the results and parameters (if you wish to see the parameters in more detail - open - analysis_df.csv).  This will sort the results in order of highest - validation set - AUC/R2 Score (You can also use it to sort by loss).  IF KFold is activated, then the standard deviation of the AUC/R2 score will be displayed as well. 'analysis_df' is a pandas dataframe, which makes it easy to sort by model_type and various other features.

```
analysis_DF.sort_values(['Valid Auc(C)-R2(R)','Train Auc(C)-R2(R)'], ascending = False)
```
<img src="docs/pictures/program_dataframe.png" align="center" height="450">


## Run Previous Model
To run a previous model from a saved dataframe, insert any of the index values into the ## Insert Index Here ## area and change *use_previous_model* to False.

```
if use_previous_model:
    params, model_selector, model_selector_mod = getSavedParams(path, load_index = ## Insert Index Here ##)
    models_to_be_created = {model_selector : 1}
```

## How To Add New Algorithms To Program
Adding new algorithms is easy for this program with this 2 step approach.  Each algorithm has the same arguments for their function and has a classification and regressor (except Lasso and Ridge).

### First Step - For Loop Addition
Add into the main *for loop* an *elif* statement for the model with the *model_selector* name you prefer with the function name to be stored in *modgen_classifier_models.py* (lightGBM example shown below).

```
elif model_selector == 'lightgbm':
      params, model = getModelLightGBM(use_previous_model, params, is_classifier)
```
### Second Step - Function Addition
The function for each algorithm is stored in *modgen_classifier_models.py*.  Each function has a dictionary of parameters which are fed into the model.  If custom options are needed for classification or regressors, then you can insert it into the correct location using *is_classifier*.
```
def getModelLightGBM(use_previous_model, params, is_classifier):
    if not use_previous_model:
        params = {}
        params['boosting_type'] = getRandomFromList(['dart', 'gbdt', 'rf'])
        params['Model_type'] = 'lightgbm_' + params['boosting_type']
        params['learning_rate'] = getRandomNumber(-3,-1, random_type = 'exp_random')
        params['num_leaves'] = getRandomNumber(2,100)
        params['min_child_samples'] = getRandomNumber(2,100)
        params['max_depth'] = getRandomNumber(1,10)
        params['n_estimators'] = 10000

    if is_classifier:
        objective_ = 'binary'
        params['metric'] = ['binary_logloss','auc']
        model = LGBMClassifier(boosting_type = params['boosting_type'], num_leaves = params['num_leaves'],
                              max_depth = params['max_depth'], learning_rate = params['learning_rate'],
                              n_estimators = params['n_estimators'], objective = objective_,
                              min_child_samples = params['min_child_samples'], random_state = 0)
    else:
        objective_ = 'regression'
        params['metric'] = ['l2']
        model = LGBMRegressor(boosting_type = params['boosting_type'], num_leaves = params['num_leaves'],
                              max_depth = params['max_depth'], learning_rate = params['learning_rate'],
                              n_estimators = params['n_estimators'], objective = objective_,
                              min_child_samples = params['min_child_samples'], random_state = 0)
    return params, model
```
### Random Number / String Selection for Models
There are 2 main functions for randomly selecting parameters for models.

#### getRandomFromList(list)
This function will return one random value from your list (can use numbers as well)
```
getRandomFromList(['dart', 'gbdt', 'rf'])
```
#### getRandomNumber(lower_range_limit, upper_range_limit, random_type = 'int')
random_type = 'int' returns a random int between lower_range_limit and upper_range_limit.
**Example:** will return (inclusive) a single *int* between 2 and 100
```
getRandomNumber(2,100)
```
random_type = 'float' returns a random float between lower_range_limit and upper_range_limit (Do not use for < 0.1)
**Example:** will return (inclusive) a single *float* between 0.5 and 2.5
```
getRandomNumber(0.5,2.5, random_type = 'float')
```
random_type = 'exp' returns a random float with exponential power.  All numbers will be 1^n.  n = range(lower_range_limit, upper_range_limit)
**Example:** will return (inclusive) a single value between 1e-13 and 1e3
```
getRandomNumber(-12,3, random_type = 'exp')
```
random_type = 'exp_random' returns a value similar to 'exp', however the 1 in (1^n) is also randomized between 1-9
**Example:** will return (inclusive) a single value between (1-9)e-13 and (1-9)e3
```
getRandomNumber(-12,3, random_type = 'exp_random')
```
