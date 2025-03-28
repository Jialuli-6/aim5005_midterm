#import libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


#In terminal, I use python3 to initiate the .py file by first typing python3 then follows the complete directory of the .py file


#load dataset
#The code and dataset are uploaded to github in one folder, directly importng dataset will be workable if use the folder terminal
df = pd.read_csv('accident.csv')
df.head()

#clean dataset remove the data without data 'Gender'
df['Gender'].isna().sum()                                           #1 missing
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)           #fillna with most frequent value         

#clean dataset fill the missing data with median 'Speed_of_Impact'
df['Speed_of_Impact'].isna().sum()                                                            #3 missing
df['Speed_of_Impact'] = df['Speed_of_Impact'].fillna(df['Speed_of_Impact'].median())          #fillna with median

#clean dataset encode categorical data 'Gender', 'Helmet_Used', 'Seatbelt_Used', change to 1 and 0
df['Helmet_Used'] = df['Helmet_Used'].map({'Yes': 1, 'No': 0})                    #set 'Yes' as 1
df['Seatbelt_Used'] = df['Seatbelt_Used'].map({'Yes': 1, 'No': 0})                #set 'Yes' as 1
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})                         #set 'Male' as 1
df.head()

#spliting the dataset
X = df[['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used']]
y = df['Survived']

#spliting the dataset into train and test with test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#count the dataset
df.count()      #199 data points after clearing one without Gender

#scale the data using standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#######################################################################################################################

#defining three models with three hyperparameters, and evaluation scores

RandomForest = RandomForestClassifier(random_state=42)  
params_r= {'n_estimators': [50, 100, 200],                  #The number of trees in the forest.
           'max_depth': [None, 10, 20],                     #The maximum depth of the tree
           'min_samples_split': [2, 5, 10],                 #The minimum number of samples required to split an internal node
           'criterion': ['gini', 'entropy', 'log_loss']     #The function to measure the quality of a split
          }

LogisticRegression = LogisticRegression()
params_l = {'C': [0.1, 1, 10],                              #Inverse of regularization strength
            'solver': ['lbfgs', 'liblinear', 'saga'],       #Algorithm to use in the optimization problem
            'tol': [0.01, 0.0001, 0.00001],                 #Tolerance for stopping criteria
            'max_iter': [100, 500, 1000]                    #Maximum number of iterations taken for the solvers to converge
           }

KNN = KNeighborsClassifier()
params_k = {'n_neighbors': [3, 5, 10],                      #Number of neighbors to use by default for kneighbors queries.
            'weights': ['uniform', 'distance'],             #Weight function used in prediction
            'metric': ['euclidean', 'cosine', 'manhattan'], #Metric to use for distance computation
            'leaf_size' : [10, 30, 100]                     #Leaf size passed to BallTree or KDTree, affect the speed 
           }

scorers = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score),
           'mcc': make_scorer(matthews_corrcoef)
          }


##################################################################################################################

#train the random forest model and predict results with the best parameters
grid_search_rf = GridSearchCV(estimator=RandomForest, param_grid=params_r, cv=5, n_jobs=-1, scoring=scorers, refit='accuracy')

# Fit GridSearchCV to the training data
y_pred_rf = grid_search_rf.fit(X_train_scaled, y_train)

# Best hyperparameters and performance
print("Best rf hyperparameters:", grid_search_rf.best_params_)

# Best model performance on the test set
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)

# Evaluate the best model on the test set using accuracy, precision, recall, f1, and mcc
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
mcc = matthews_corrcoef(y_test, y_pred_rf)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")


#please see results
#Best hyperparameters: {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'n_estimators': 50}
#Test Accuracy: 0.5000
#Test Precision: 0.4286
#Test Recall: 0.3333
#Test F1 Score: 0.3750
#Matthews Correlation Coefficient (MCC): -0.0316


###################################################################################################################

#train the logistic regression model and predict results with the best parameters
grid_search_lr = GridSearchCV(estimator=LogisticRegression, param_grid=params_l, cv=5, n_jobs=-1, scoring=scorers, refit='accuracy')

# Fit GridSearchCV to the training data
grid_search_lr.fit(X_train_scaled, y_train)

# Best hyperparameters and performance
print("Best lr hyperparameters:", grid_search_lr.best_params_)

# Best model performance on the test set
best_lr = grid_search_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)

# Evaluate the best model on the test set using accuracy, precision, recall, f1, and mcc
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
mcc = matthews_corrcoef(y_test, y_pred_lr)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

#please see results
#Best hyperparameters: {'C': 0.1, 'max_iter': 100, 'solver': 'liblinear', 'tol': 0.01}
#Test Accuracy: 0.5250
#Test Precision: 0.4667
#Test Recall: 0.3889
#Test F1 Score: 0.4242
#Matthews Correlation Coefficient (MCC): 0.0259


###################################################################################################################

#train the KNN model and predict results with the best parameters
grid_search_k = GridSearchCV(estimator=KNN, param_grid=params_k, cv=5, n_jobs=-1, scoring=scorers, refit='accuracy')

# Fit GridSearchCV to the training data
y_pred = grid_search_k.fit(X_train_scaled, y_train)

# Best hyperparameters and performance
print("Best KNN hyperparameters:", grid_search_k.best_params_)

# Best model performance on the test set
best_k = grid_search_k.best_estimator_
y_pred_k = best_k.predict(X_test_scaled)

# Evaluate the best model on the test set using accuracy, precision, recall, f1, and mcc
accuracy = accuracy_score(y_test, y_pred_k)
precision = precision_score(y_test, y_pred_k)
recall = recall_score(y_test, y_pred_k)
f1 = f1_score(y_test, y_pred_k)
mcc = matthews_corrcoef(y_test, y_pred_k)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")


#please see results
#Best hyperparameters: {'leaf_size': 10, 'metric': 'cosine', 'n_neighbors': 3, 'weights': 'uniform'}
#Test Accuracy: 0.5000
#Test Precision: 0.4375
#Test Recall: 0.3889
#Test F1 Score: 0.4118
#Matthews Correlation Coefficient (MCC): -0.0205

#I use accuracy as the metric to evaluate the performance of the model, and it turns out that logistic regression is the better model to use for this prediction.


######################################################################################################################

#print the results of logistic regression
# Get feature names
feature_names = X_train.columns

# Get coefficients from ligistic regression with best hyperparameters
coefficients = grid_search_lr.best_estimator_.coef_[0]

# Create a DataFrame
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

print(coef_df)


#please see results fron logistic regression
#           Feature  Coefficient
#1           Gender     0.283156
#0              Age     0.123963
#4    Seatbelt_Used     0.090643
#2  Speed_of_Impact    -0.024509
#3      Helmet_Used    -0.100673
