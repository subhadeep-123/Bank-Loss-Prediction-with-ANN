
# coding: utf-8

# In[1]:


#Uploading the file
from google.colab import files
uploaded = files.upload()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[4]:


df = pd.read_csv('Churn_Modelling.csv')

print("Shape:", df.shape)


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


#Checking the no of 0 bacl accounts
empty_balc = []
for i in df['Balance']:
  if i == 0:
    empty_balc.append(i)
  else:
    pass
print(len(empty_balc))


# In[10]:


df['Geography'].value_counts()


# In[11]:


df['Gender'].value_counts()


# In[12]:


#Encoding Male and Female to 1 and 0
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
df['Gender'].head(5)


# In[13]:


#Spliting into X and Y
X = df.iloc[:, 3:13].values
Y = df.iloc[:, 13].values

print("X: {}".format(X.shape))
print("Y: {}".format(Y.shape))


# In[14]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Creating instance of LabelEncoder class
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#Creating instance of OneHotEncoder class
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.2,
                                                    random_state = 0)


# In[ ]:


#Standar Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[17]:


#Building our baseline dummy classifier
from sklearn.dummy import DummyClassifier
clf = DummyClassifier()
clf.fit(X_train, y_train)

#Predicting Results
y_pred = clf.predict(X_test)

#Calculating Resulta
print("CM: \n",confusion_matrix(y_test, y_pred))
print("acc: {0}%".format(accuracy_score(y_test, y_pred) * 100))


# In[18]:


import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# ## Building Our Artificial Neural Network

# In[19]:


# Initialising the ANN
classifier = Sequential()

#Input and 1st Hidden Layer
classifier.add(Dense(units = 13,
                     activation = 'relu',
                     kernel_initializer = 'uniform',
                     input_dim = 11))
classifier.add(Dropout(p = 0.2))


#2nd Hidden Layer
classifier.add(Dense(units = 13,
                     activation = 'relu',
                     kernel_initializer = 'uniform'))
classifier.add(Dropout(p = 0.2))   


#3rd Hidden Layer
classifier.add(Dense(units = 13,
                     activation = 'relu',
                     kernel_initializer = 'uniform'))
classifier.add(Dropout(p = 0.2))               

#Output Layer
classifier.add(Dense(units = 1,
                     activation = 'sigmoid',
                     kernel_initializer = 'uniform'))
               
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])               


# In[20]:


classifier.summary()


# In[21]:


#training our ANN Model
history = classifier.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_split=0.25)


# In[22]:


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[27]:


#Model Evaluation

print('Accuracy Score: ' + str(accuracy_score(y_test, y_pred)))

print('Precision Score: ' + str(precision_score(y_test, y_pred)))

print('Recall Score: ' + str(recall_score(y_test, y_pred)))

print('F1 Score: ' + str(f1_score(y_test, y_pred)))

print('Classification Report: \n' + str(classification_report(y_test, y_pred)))


# In[33]:


#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


# Evaluating the ANN with KFold Cross Validation

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu', input_dim = 11))
    
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu'))
    
    classifier.add(Dense(units = 1, 
                         kernel_initializer = 'uniform', 
                         activation = 'sigmoid'))
    
    classifier.compile(optimizer = 'adam', 
                       loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
    
    return classifier
  
classifier = KerasClassifier(build_fn = build_classifier, 
                             batch_size = 10, 
                             epochs = 100)

accuracies = cross_val_score(estimator = classifier, 
                             X = X_train, 
                             y = y_train, 
                             cv = 10, 
                             n_jobs = -1)


# In[36]:


mean = accuracies.mean()
print("Mean: ", mean)

variance = accuracies.std()
print('Variance: ', variance)


# In[ ]:


# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu', input_dim = 11))
    
    classifier.add(Dense(units = 6, 
                         kernel_initializer = 'uniform', 
                         activation = 'relu'))
    
    classifier.add(Dense(units = 1, 
                         kernel_initializer = 'uniform', 
                         activation = 'sigmoid'))
    
    classifier.compile(optimizer = optimizer, 
                       loss = 'binary_crossentropy', 
                       metrics = ['accuracy'])
    
    return classifier
  
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [100, 200],
              'optimizer': ['adam', 'rmsprop']}

random_search = RandomizedSearchCV(estimator = classifier,
                                   param_distributions  = parameters,
                                   n_iter = 15,
                                   cv = 10,
                                   n_jobs = -1)

random_search = random_search.fit(X_train, y_train)


# In[ ]:


best_parameters = random_search.best_params_
print(best_parameters)

best_accuracy = random_search.best_score_
print(best_accuracy)

