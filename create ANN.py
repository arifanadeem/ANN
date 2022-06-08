# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

a=np.array([[1,2,3],[4,5,6]])
print(a)

datasets= pd.read_csv("C:\\Users\\arifa\\Downloads\\Complete-Deep-Learning-master\\Complete-Deep-Learning-master\\ANN\\Churn_Modelling.csv")
print(datasets.head())

# slicing the dataset as independent and dependent variable
X= datasets.iloc[:,3:13]
y= datasets.iloc[:,13]

# create dummy variables
geography=pd.get_dummies(datasets["Geography"])
gender= pd.get_dummies(datasets["Gender"])

X= X.drop(["Geography","Gender"],axis=1)
X= pd.concat([X,gender,geography],axis=1)

# training and testig dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling 

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)


#X_train[X_train < 0] = 0

#X_test[X_test < 0] = 0

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


classifier= Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 13))
classifier.add(Dropout(0.3))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
classifier.add(Dropout(0.3))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
classifier.add(Dropout(0.3))
# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)




######### hyperparameter for binary classification ########


from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid
from keras.layers import LeakyReLU



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(1)) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)




layers = [[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)


[grid_result.best_score_,grid_result.best_params_]
pred_y = grid.predict(X_test)
y_pred = (pred_y > 0.5)
y_pred


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
score



