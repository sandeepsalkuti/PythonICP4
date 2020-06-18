#importing pandas library so we can read csv file and perform other operations
import pandas as pd

# reading csv file
traindata=pd.read_csv("train.csv")

#mapping categorical values to numerical values as integers
traindata['Sex']=traindata['Sex'].map({'female':1,'male':0}).astype(int)

#using corr() to correlate Survived column against Sex column and printing correlation value
print(traindata['Survived'].corr(traindata['Sex']))