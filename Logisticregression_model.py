# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn import metrics
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


'''----------------------------------------------Regression model-----------------------------------------------------------'''


dataset = pd.read_csv('creditcard.csv')
 
X = dataset.iloc[:, :-1]
 
y = dataset.iloc[:, -1]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
# Using Skicit-learn to split data into training and testing sets 
from sklearn.model_selection import train_test_split 
# Split the data into training and testing sets 
xTrain, xTest, yTrain, yTest = train_test_split(scaled_X,y, test_size = 0.3, random_state = 42)


from imblearn.over_sampling import ADASYN 
from collections import Counter
ada = ADASYN(random_state=42)
print('Original dataset shape {}'.format(Counter(yTrain)))
x_os_Train, y_os_Train = ada.fit_sample(xTrain, yTrain)
print('Resampled dataset shape {}'.format(Counter(y_os_Train)))


from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(x_os_Train,y_os_Train)
yPred=regressor.predict(xTest)
accuracy=accuracy_score(yTest,yPred)
print('Accuracy= {}'.format(accuracy))

# Saving model to disk
pickle.dump(regressor, open('Logisticregression_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('Logisticregression_model.pkl','rb'))
print(model.predict([[72824,-1.111495232,-0.257575207,2.250209635,1.152670903,0.432904474,1.254126028,-0.58416279,-0.609681605,1.014602463,0.334532824,0.826374844,0.1968869,-1.885993498,-0.472025637,-0.578141396,-1.243007101,0.570459827,-0.159056766,0.407187527,-0.510613592,0.86291279,0.927825035,-0.343058086,-0.25626823,-0.600742166,-0.180331288,0.026762226,-0.358335321,45.03]]))
