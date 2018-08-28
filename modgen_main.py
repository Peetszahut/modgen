
# coding: utf-8

# In[1]:


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


# In[6]:


### Titanic Classifier Testing Set

path = "/Users/jeromydiaz/Desktop/all/"
train = pd.read_csv("/Users/jeromydiaz/Desktop/all/train.csv")
test = pd.read_csv("/Users/jeromydiaz/Desktop/all/test.csv")

def titleCheck(x):
    # x[0] = name
    # x[1] = pclass
    important_title = ['Rev.', 'Dr.', 'Mme', 'Major', 'Lady', 'Sir.', 'Col.', 'Capt.', 'Countess']
    for title in important_title:
        if title in x[0] and x[1] == 1:
            return 1 
    return 0

##### Try Me 
test.fillna(test[test.Pclass == 3].Fare.mean(), inplace = True)

train['Under18P1'] = np.where((train['Age'] < 18) & (train['Pclass'] == 1), 1, 0)
train['Over60'] = np.where((train['Age'] > 60), 1, 0)
train['Alone'] = np.where((train['Parch'] == 0) & (train['SibSp'] == 0), 1, 0)
train['FareOver50'] = np.where((train['Fare'] > 50), 1, 0)
train['1stSpecial'] = train[['Name','Pclass']].apply(titleCheck, axis = 1)

test['Under18P1'] = np.where((test['Age'] < 18) & (test['Pclass'] == 1), 1, 0)
test['Over60'] = np.where((test['Age'] > 60), 1, 0)
test['Alone'] = np.where((test['Parch'] == 0) & (test['SibSp'] == 0), 1, 0)
test['FareOver50'] = np.where((test['Fare'] > 50), 1, 0)
test['1stSpecial'] = test[['Name','Pclass']].apply(titleCheck, axis = 1)

# For Train
y_train = train['Survived']
x_train = train.drop(['PassengerId','Survived','Name','Ticket','Cabin','Embarked', 'Parch', 'SibSp'], axis=1)
encSex = LabelEncoder()
testEnc = encSex.fit_transform(x_train.Sex)
x_train['Sex'] = testEnc
x_train['Age'].fillna(x_train.Age.mean(), inplace = True)

# For Test
x_test = test.drop(['PassengerId','Name','Ticket','Cabin','Embarked', 'Parch', 'SibSp'], axis=1)
x_test['Sex'] = encSex.fit_transform(x_test.Sex)
x_test['Age'].fillna(x_test.Age.mean(), inplace = True)


# In[8]:


######## Model Options ########
### K-Fold Options
# 'normal', 'strat', 'normal_repeat', 'strat_repeat' - (type, # repeats)
kfold_update = True
kfold = 4
kfold_distribution = 'normal'
kfold_repeats = 1

### Scaler Option: StandardScaler(), Normalizer(), MinMaxScaler(), RobustScaler()
# If scaler_select = None, then no scaling will be done
scaler_select = StandardScaler() 

### Model creation options
# If previousModel = False: Use new modelCreation dictionary to make models, else, use previous index to get parameters
# If train_full = True: Train data on test set and make submission file of results
# If ensemble = True: Ensemble all previous index models together
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
    params, modelSelection, modelSelectionMod = getSavedParams(load_index = 150)
    modelCreation = {modelSelection : 1}
else:
    modelCreation = {
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
if previousModel: kfold_update = False  
if kfold_update: 
    kfold_type = (kfold_distribution, kfold_repeats) 
else: 
    kfold = 1
    kfold_type = None

### Scaler Function Inputs
if scaler_select: 
    X_train, X_test = scaleSelector(x_train, x_test, scaler_select)
else:
    X_train, X_test = x_train, x_test
Y_train, X_valid, Y_valid = y_train, None, None

# If kfold == 1: Split the data using train_test_split
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
        
        elif modelSelection == 'neuralnetwork':
            params, model = getModelNeuralNetwork(previousModel, params)
            
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
                                                                    X_test, params, train_full, model, kfold = kfold,
                                                                    kfold_type = kfold_type,
                                                                    modeltype = modelSelection)
        
        analysisDF = dataFrameUpdate(params, Y_train, Y_valid, Pred_train, Pred_valid, analysisDF, kfold_DF,
                                    kfold_update)
        if ensemble: ensembleList.append(Pred_test)
    

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


# In[ ]:


analysisDF.sort_values(['Valid Auc','Train Auc'], ascending = False)


# In[10]:


test1 = None
if test1:
    print('yes')
else:
    print('no')

