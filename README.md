# Modgen - Automatic Model Generator
## Overview of Modgen
This program was created for rapid feature engineering without the need to optimize each model.  Modgen is designed to develop a quick overview of how your updated features will react to each model.  You can use one specific algorithm or a wide variety (depending on your interests) with a random feature range which can be easily changed at anytime by the user.

## Libraries Used
* Pandas
* Numpy
* Matplotlib
* SkLearn
* Keras
* LightGBM
* XGBoost
* tqdm

## How It Works
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
