# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:25:43 2019

@author: deepikapantola
"""

# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print (weather_encoded)


# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
temp_encoded
label=le.fit_transform(play)
print ("Temp:",temp_encoded)
print ("Play:",label)

#Combinig weather and temp into single listof tuples
features1=list(zip(weather_encoded,temp_encoded))
features1

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features1,label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)



predicted= model.predict([[2,1]]) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)

####Example 2

from sklearn import datasets

#Load dataset
wine=datasets.load_wine()
wine
len(wine['data'])
# print the names of the 13 features
print ("Features: ", wine.feature_names)

# print the label type of wine(class_0, class_1, class_2)
print ("Labels: ", wine.target_names)


# print data(feature)shape
wine.data.shape

# print the wine data features (top 5 records)
print (wine.data[0:5])

# print the wine labels (0:Class_0, 1:class_2, 2:class_2)
print (wine.target)

# Import train_test_split function

from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) # 70% training and 30% test


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)
y_pred
##Evaluating Model

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


