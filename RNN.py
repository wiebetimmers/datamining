#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, SimpleRNN
from keras import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
import math
import plotly 
import plotly.express as px
import plotly.graph_objects as go
#from google.colab import drive
#drive.mount('/content/drive')


# In[3]:


df = pd.read_csv("dataset_mood_smartphone.csv")


# In[4]:


# Create columns for variables that have a value
for var in df["variable"].unique():
    df['%s'%var] = np.where(df['variable'] == var, df['value'], np.NaN)

# Split the date-time in seperate columns, transform to a datetime format
df[['date', 'time']] = df['time'].str.split(' ', 1, expand=True)
df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S.%f')


# In[5]:


# Pick one sample user
df1 = df[df['id'] == 'AS14.01']
df_avg = pd.DataFrame()

# Average over the day
take_mean = set(("mood", "activity", "circumplex.arousal", "circumplex.valence"))

# Total time everyday
take_sum = set(("screen", "appCat.builtin","appCat.communication","appCat.entertainment", 
                "appCat.finance", "appCat.game", "appCat.office", "appCat.other", "appCat.social",  
                "appCat.travel", "appCat.unknown", "appCat.utilities", "appCat.weather"))

# Amount of calls or texts a day
take_count = set(("call", "sms"))

for var in df["variable"].unique():
    df2 = df1[df1[var].notna()]
    if var in take_mean:
        df_mean = df2.groupby('date').mean()
        df_avg[var] = df_mean[var]
    if var in take_sum:
        df_sum = df2.groupby('date').mean()
        df_avg[var] = df_sum[var]
    if var in take_count:
        df_count = df2.groupby('date').count()
        df_avg[var] = df_count[var]


df_avg


# In[6]:


arima_df = df_avg[2:]
arima_df = arima_df.drop('appCat.weather', 1)
arima_df = arima_df.drop('appCat.unknown', 1)
arima_df = arima_df.drop('appCat.game', 1)
arima_df = arima_df.drop('appCat.office', 1)
print(arima_df)


# In[7]:


def prep_data(datain, time_step):
    # 1. y-array  
    # First, create an array with indices for y elements based on the chosen time_step
    y_indices = np.arange(start=time_step, stop=len(datain), step=time_step)
    # Create y array based on the above indices 
    y_tmp = datain[y_indices]
    y_tmp = y_tmp[:, 0]
    print(y_tmp)
    #y_tmp = datain[['mood']]
    # 2. X-array  
    # We want to have the same number of rows for X as we do for y
    rows_X = len(y_tmp)
    # Since the last element in y_tmp may not be the last element of the datain, 
    # let's ensure that X array stops with the last y
    X_tmp = datain[range(time_step*rows_X)]
    X_tmp = X_tmp[:, 1:]
    print(X_tmp)
    # Now take this array and reshape it into the desired shape
    X_tmp = np.reshape(X_tmp, (rows_X, time_step, 2))
    return X_tmp, y_tmp


# In[8]:


labels = ['mood', activity, screen]


# In[ ]:


##### Step 1 - Select data for modeling and apply MinMax scaling
X=arima_df[['mood', 'activity', 'screen']]
scaler = MinMaxScaler((0, 1))
X_scaled=scaler.fit_transform(X)


##### Step 2 - Create training and testing samples
train_data, test_data = train_test_split(X_scaled, test_size=0.2, shuffle=False)


##### Step 3 - Prepare input X and target y arrays using previously defined function
time_step = 5
X_train, y_train = prep_data(train_data, time_step)
X_test, y_test = prep_data(test_data, time_step)


##### Step 4 - Specify the structure of a Neural Network
model = Sequential(name="First-RNN-Model") # Model
model.add(Input(shape=(time_step,2), name='Input-Layer')) # Input Layer - need to speicfy the shape of inputs
model.add(SimpleRNN(units=1, activation='tanh', name='Hidden-Recurrent-Layer')) # Hidden Recurrent Layer, Tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))
model.add(Dense(units=1, activation='tanh', name='Hidden-Layer')) # Hidden Layer, Tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))
model.add(Dense(units=1, activation='linear', name='Output-Layer')) # Output Layer, Linear(x) = x


##### Step 5 - Compile keras model
model.compile(optimizer='adam', # default='rmsprop', an algorithm to be used in backpropagation
              loss='mean_squared_error', # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
              metrics=['MeanSquaredError', 'MeanAbsoluteError'], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
              loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
              weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
              run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
              steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
             )


##### Step 6 - Fit keras model on the dataset
model.fit(X_train, # input data
          y_train, # target data
          batch_size=1, # Number of samples per gradient update. If unspecified, batch_size will default to 32.
          epochs=20, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
          verbose='auto', # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
          callbacks=None, # default=None, list of callbacks to apply during training. See tf.keras.callbacks
          validation_split=0.0, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 
          #validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch. 
          shuffle=True, # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
          class_weight=None, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
          sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
          initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
          steps_per_epoch=None, # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. 
          validation_steps=None, # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
          validation_batch_size=None, # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
          validation_freq=1, # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
          max_queue_size=10, # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
          workers=1, # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
          use_multiprocessing=False, # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. 
         )


##### Step 7 - Use model to make predictions
# Predict the result on training data
pred_train = model.predict(X_train)
# Predict the result on test data
pred_test = model.predict(X_test)


##### Step 8 - Model Performance Summary
print("")
print('-------------------- Model Summary --------------------')
model.summary() # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
print("Note, the last parameter in each layer is bias while the rest are weights")
print("")
for layer in model.layers:
    print(layer.name)
    for item in layer.get_weights():
        print("  ", item)
print("")
print('---------- Evaluation on Training Data ----------')
print("MSE: ", mean_squared_error(y_train, pred_train))
print("")

print('---------- Evaluation on Test Data ----------')
print("MSE: ", mean_squared_error(y_test, pred_test))
print("")


# In[ ]:


print(y_train)
print(pred_train)


# In[ ]:




