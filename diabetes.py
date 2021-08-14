import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import category_encoders as ce
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras



# Importing dataset
dataset = pd.read_csv('diabetes.csv')

# To show more columns in console
pd.set_option('display.max_columns', 10)

# Dividing dataset in  train and test set

x = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)



def data_distplot(data):
    """
    Function to print dist plot of a data set
    :param data: dataset
    :return:
    """
    for cols in data.columns:
        plt.figure()
        sns.distplot(data[cols])


def data_boxplot(data):
    """
    Function to print scatter plot of a data set
    :param data: dataset
    :return:
    """
    for cols in data.columns:
        plt.figure()
        sns.boxplot(y=data[cols], x=data['Outcome'], hue=data['Outcome'])


def data_set_exploration(data):
    """
    Exploring data set for insights
    :param data: dataset
    :return: None
    """
    print('Looking at data \n :', data.head())
    print('Numeric insight about dataset \n', data.describe())
    print('Information of data \n', data.info())
    # By looking at data numerically it is clear that
    # Glucose and Insulin have high variability
    # Blood pressure and skin thickness also have more variance
    # Non of the columns are null

    # Let's Visualize data to gain further insights

    #data_distplot(data)

    # From visualization there is some insights we have gained
    # AGE : age is right skewed and have peak between 20 and 30
    # BMI : Bmi is bimodal distribution with 0 Bmi might be missing data or outliers
    # INSULIN : It is highly concentrated toward 0 is right skewed
    # SKIN THICKNESS : Bimodal distribution with lots of reading concentrated at 0
    # Glucose : It is also somewhat bimodal with some reading at zero but mostly at 100 slightly left skewed
    # PREGNANCIES :  Right Skewed with greater concentration around zero and mean at around 3
    # DIABETES PEDIGREE FUNCTION : Right skewed greater than zero
    # Diabetes pedigree function (a function which scores likelihood of diabetes based on family history

    # co-relational matrix
    corr_matrix = data.corr()
    print(corr_matrix['Outcome'])
    plt.figure()
    sns.heatmap(corr_matrix, vmax=0.3, center=0, square=True)
    # from correlation matrix it is clear that outcome is mostly dependent on :
    # Glucose, BMI, Age and pregnancies
    # Glucose is highly dependent on insulin, Age and Bmi
    # BMI is dependent on Blood pressure, Skin thickness and insulin
    # Age is not highly dependent but affects glucose and blood pressure
    # Pregnancies is highly dependent on Age*

    # Checking unique values and trying to understand more about 0 in labels
    print('Total unique values in data \n', data.nunique())
    # Unique value columns wise
    print('Unique cols \n', [pd.value_counts(data[cols]) for cols in data.columns], '\n')
    # Pregnancies : zero is not an issue since women can never be pregnant
    # Glucose : 4 people with zero which mean these people did not tested glucose level
    # Bmi : 11 cases of 0 bmi which can not happen since not tested
    # Blood Pressure : 35 non tested values
    # Skin Thickness : 227 non tested values
    # Insulin : 374 non tested

    # Identifying Outliers
    #data_boxplot(data)


data_set_exploration(dataset)


# Our dataset exploration is completed we have gained useful insights from data
# Now, we will do some feature engineering to get more info.


def feature_engineering(data):
    """
    To clean and add new features in dataset for better predictions
    :param data: dataset to be cleaned and optimized
    :return: cleaned and optimized dataset
    """

    # Finding and removing outliers via IQR
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3 - q1
    print(IQR)
    #print((data < (q1 - IQR)) |(data > (q3 +  IQR)))
    # Removing outliers
    data = data[~((data < (q1 - 1.5*IQR)) |(data > (q3 + 1.5*IQR))).any(axis=1)].copy()
    print(data.shape)
    #data_boxplot(data)
    # Dealing with missing values,
    # we can replace missing of insulin with mean for better predictions s it have more zeros.
    data['Insulin'].replace(to_replace=0, value=data['Insulin'].mean(), inplace=True)
    for cols in data.columns:
        data[cols].replace(to_replace=0, value=data[cols].median(), inplace=True)

    # Adding new features
    # Young, middle-aged and old adults

    data['AgeGroup'] = data['Age'].apply(lambda x: 'Young' if x < 38 else ('Mid-aged' if 35 <= x <= 58 else 'Old'))
    data['BMIGroup'] = data['BMI'].apply(lambda x: 'Under-weight' if x < 18.5 else ('Normal' if 18.5 <= x <= 24.9 else 'Over-weight'))
     # GTT result used here
    data['GlucoseGroup'] = data['Glucose'].apply(lambda x: 'Non-diabetic' if x < 110 else ('Impaired-glucose tolerance' if 110 <= x <= 126 else 'Diabetic'))
    data['BloodPressureGroup'] = data['BloodPressure'].apply(lambda x: 'Normal' if x < 80 else ('High-stage1' if 80 <= x <= 89 else ('High-stage2' if 90 <= x <= 120 else 'Hyperintensive-crisis')))

    data['Glucose'] = np.log1p(data['Glucose'])
    data['Insulin'] = np.log1p(data['Insulin'])
    data['BloodPressure'] = np.log1p(data['BloodPressure'])
    data['BMI'] = np.log1p(data['BMI'])
    data['SkinThickness'] = np.log1p(data['SkinThickness'])
    data['Age'] = np.log1p(data['Age'])
    data['DiabetesPedigreeFunction'] = np.log1p(data['DiabetesPedigreeFunction'])
    data['Pregnancies'] = np.log1p(data['Pregnancies'])
    return data

# Dividing dataset in  train and test set
dataset = feature_engineering(dataset.copy())
print('New shape of dataset after removing outliers', dataset.shape)
x = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


# Category encoding
cat_enc = ce.OneHotEncoder()
# Standardizing data
sc = StandardScaler()
# Pipeline for cat encoder and standard scaler
pipe = Pipeline(steps=[('cat_enc', cat_enc), ('sc', sc)])
x_train_scaled = pipe.fit_transform(x_train)
x_test_scaled = pipe.transform(x_test)
print('Scaled x_train', x_train_scaled.shape)

# Models


xgboost = xgb.XGBClassifier(objective='binary:logistic')
learning_rate = [0.3, 0.1, 0.09, 0.07]
n_estimator = [100, 150]
max_depth = [2, 4]
cv = [1, 1.5, 2]
param_xgb = dict(max_depth=max_depth, learning_rate=learning_rate,
                 n_estimator=n_estimator, cv=cv)


def roc_scoring(estimator, param):

    print('----------Executing Grid-----------')
    grid = GridSearchCV(estimator, param, cv=5, n_jobs=-1)
    grid.fit(x_train_scaled, y_train)
    print('Best Model', grid.best_params_)
    model = grid.best_estimator_
    predictions = cross_val_predict(model, x_test_scaled, y_test, cv=5)
    print(confusion_matrix(y_test, predictions))
    score = np.mean(cross_val_score(model, x_test_scaled, y_test, cv=5, scoring='roc_auc'))
    print(np.around(score, decimals=4))
    return model


xgb_model = roc_scoring(estimator=xgboost, param=param_xgb)

plt.show()
