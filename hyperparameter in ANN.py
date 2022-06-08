# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:49:01 2022

@author: arifa
"""

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

df1=pd.read_csv("Real_Combine.csv")
X1= df1.iloc[:, :-1]
y1=df1.iloc[:,-1]

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 22)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=30,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model





tuner = RandomSearch(
    build_model,
    objective='val_mean_absolute_error',
    max_trials=5,
    executions_per_trial=3,
    directory='project',
    project_name='Air Quality Index')



tuner.search_space_summary()



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=0)
tuner.search(X_train, y_train,
             epochs=5,
             validation_data=(X_test, y_test))




tuner.results_summary()


