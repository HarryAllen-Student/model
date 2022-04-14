import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('https://raw.githubusercontent.com/andvise/DataAnalyticsDatasets/main/dm_assignment2/sat_dataset_train.csv')
df.shape
df.info()
df.head()

# Check if there are any NaN or infintie values in the dataset
def check_nan_and_infinite_values(df):
    print('NaN values in dataset:', np.any(np.isnan(df)))
    print('Inf values in dataset:', not np.all(np.isfinite(df)))

check_nan_and_infinite_values(df)

# Since there are NaN anf infinite values, we replace them with 0. 
# This is necessary to do as otherwise the model will throw an error when we start training 

df = df.replace([np.inf, -np.inf, np.nan], 0)

# Hence, no more NaN and inf values. 
check_nan_and_infinite_values(df)

# SVM model using default parameters

# Splitting dataset into train, test with 70, 30 ratio
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']) , df['target'], test_size=0.3)

# Declaring default SVC classifier model
svm = SVC()

# Fitting classifier with training data
svm.fit(X_train, y_train)

# Performing predictions with test data
y_preds = svm.predict(X_test)
print('Accuracy for SVM model with default parameters:', accuracy_score(y_test, y_preds))

# Feature Normalization using `Standard Scaling`

# Copying df into a new variable
df2 = df

# Scaling all columns of df except `target` column 
df2 = StandardScaler().fit_transform(df2.drop(columns=['target']))

# Convert the scaled df from dtype array to dtype dataframe
df2 = pd.DataFrame(df2, columns=[df.columns.drop('target')])

df2.shape

# df with scaled features
df2.head()

# Hold-out method for splitting normalized df
X_train2, X_test2, y_train2, y_test2 = train_test_split(df2, df['target'], test_size=0.3)

# Feature Reduction using PCA

# Retaining 90% of explained variance
pca = PCA(n_components=0.9)   

# Extracting the columns onto the scaled 
pca.fit(df2)

# Transforming the train set  
X_train2 = pca.transform(X_train2)

X_train2.shape

# Transforming the test set 
X_test2 = pca.transform(X_test2)

X_test2.shape

#SVM using Cross-Validation and HyperParameter tuning

# Describing hyperparameters for Grid Search ## Takes roughly 1min 
param_grid = { 
    'C':[0.1,1,100],
    'kernel':['rbf','linear'],
    'degree':[1,3,5],
    'gamma': [0.001, 0.0001]
}
# Declaring GridSearch with appropriate model and parameters
grid = GridSearchCV(SVC(),param_grid,  scoring='accuracy', return_train_score=False, verbose=1)

# Applying Grid Search and Cross Validation
grid_search = grid.fit(X_train2, y_train2)

# Check the best parameters found
grid_search.best_params_

# Check the best score achieved with the best parameter
grid_search.best_score_

# Declare SVM with the best parameters found in Grid Search  
SVM_test = grid_search.best_estimator_

# Train best model with scaled training data
SVM_test.fit(X_train2, y_train2)

# Make predictions with scaled test data
y_preds2 = SVM_test.predict(X_test2) 
print('Accuracy for SVM model with Grid Search and CV:', accuracy_score(y_test2, y_preds2)) 