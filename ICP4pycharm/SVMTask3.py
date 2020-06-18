import pandas as pd                                      #importing pandas
from sklearn.model_selection import train_test_split     #importing tarin_test_split from model_selection in scikit learn
from sklearn import svm                                  #importing svm from scikit learn
from sklearn import metrics                              #importing metrics from scikit learn
from sklearn.metrics import classification_report        #importing classification_report from metrics library

data=pd.read_csv("glass1.csv")                           #reading csv file using pandas read_csv
X = data.drop("Type",axis=1)                             #taking all columns in X-axis except Type column by dropping it
y = data["Type"]                                         #taking only Type column in Y-axis
#Using train_test_split splitting data to training(70%) and testing set(30%) so we can test and predict upon the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
var=svm.SVC(kernel="linear")                             #calling SVC function from SVM model and using linear kernel
var.fit(X_train,y_train)                                 #fitting X and Y values to the model
y_pred=var.predict(X_test)                               #making predictions on test data

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100) #calculating accuracy scores for test and predicted
print("Classification report\n", classification_report(y_test,y_pred,zero_division=1)) #finding classification report