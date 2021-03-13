
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer

# Importing the training dataset
train_dataset = pd.read_csv('train_data.csv')
X_train = train_dataset.iloc[:, [2,3,4,5]].values
y_train = train_dataset.iloc[:, 6].values


# Importing the testing dataset
test_dataset = pd.read_csv('test_data.csv')
X_test = test_dataset.iloc[:, [2,3,4,5]].values
y_test = test_dataset.iloc[:, 6].values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 2] = labelencoder_X.fit_transform(X_train[:, 2])
onehotencoder = ColumnTransformer([('CodedDay', OneHotEncoder(), [2])], remainder='passthrough')
X_train = onehotencoder.fit_transform(X_train).toarray()

X_train[:, 3] = labelencoder_X.fit_transform(X_train[:, 3])

X_train[:, 4] = labelencoder_X.fit_transform(X_train[:, 4])

X_train[:, 5] = labelencoder_X.fit_transform(X_train[:, 5])

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_XY = LabelEncoder()
X_test[:, 2] = labelencoder_XY.fit_transform(X_test[:, 2])
onehotencoder = ColumnTransformer([('CodedDay', OneHotEncoder(), [2])], remainder='passthrough')
X_test = onehotencoder.fit_transform(X_test).toarray()

X_test[:, 3] = labelencoder_XY.fit_transform(X_test[:, 3])

X_test[:, 4] = labelencoder_XY.fit_transform(X_test[:, 4])

X_test[:, 5] = labelencoder_XY.fit_transform(X_test[:, 5])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import f1_score
f1s = f1_score(y_test, y_pred, average='micro') 

print("Accuracy Score:")
print(ac)
print("\nConfusion Matrix:")
print(cm)
print("\nF1 Score:")
print(f1s)
