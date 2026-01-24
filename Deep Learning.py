# Importing the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


print("TensorFlow version:", tf.__version__)

# Importing the dataset
# Download the dataset from: https://www.kaggle.com/datasets/shubh0799/churn-modelling/data?select=Churn_Modelling.csv
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)


# Concatenate the Data Frames
X=pd.concat([X,geography,gender],axis=1)

# Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part-1: Making of the ANNs
############################################################
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=11,activation='relu'))

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6,activation='relu'))

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=1,activation='relu'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=50)

# list all data in history
print(model_history.history.keys())

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Part-2: Making the predictions and evaluating the model
############################################################

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print(score)


## CNN Code can be checked from below link
# CNN: https://colab.research.google.com/drive/16h9Tqf1nj0XaE3wapmYcpLZYqbmp4uri
