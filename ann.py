"""
Created on Sat Feb 16 16:17:28 2019

@author: henrik
"""

import numpy as np # This do much of the math -> and arrays
import matplotlib.pyplot as plt 
import pandas as pd #Import data set

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])

labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

#--------------------Split to test and training set---------------------------# 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#-------------------------Feature scaling-------------------------------------#
#It's important to set the variables at the same scale. So we don't end up with a huge number that breaks
#Smaller values. As in the Data.csv data set salary will dominate age
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#------------------------------ANN MODEL--------------------------------------#
import keras
from keras.models import Sequential
from keras.layers import Dense #This initelizes all the weigths close to zero
from keras.layers import Dropout

"""
classifier = Sequential() #It's a sequense of layers

#Add first layer and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim = 11)) 

#Add secund hidden layer 
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu')) 

#Add output layer
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid', input_dim = 6)) 

#Compile the code
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'] )

classifier.fit(x_train, y_train, batch_size = 5, epochs=100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)


#Predict a single new observation

new_prediction = classifier.predict(sc_x.fit_transform(np.array([[0.0,0, 600, 1, 40, 3, 6000, 2, 1,1, 50000]])))
new_prediction = (new_prediction > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""

#K-Fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim = 11)) 
    classifier.add(Dropout(p = 0.1))
    
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))     
    classifier.add(Dropout(p = 0.1))
    
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid', input_dim = 6)) 
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'] )
    return classifier

model = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32], 
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

gid_search = GridSearchCV(estimator = model, 
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10)
gid_search = gid_search.fit(X = x_train, y = y_train)
best_parameters = gid_search.best_params_
best_accuracy = gid_search.best_score_
print(best_parameters)
print(best_accuracy)
#accuracies = cross_val_score(estimator= model, X = x_train, y = y_train, cv=10, n_jobs=-1)





