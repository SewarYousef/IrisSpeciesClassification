
# Importing the packages we going to use in this classification project
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Loading the data
data = pd.read_csv(r'C:\Users\sewar\OneDrive\Desktop\datasets\Iris.csv')
print(data.head(), '\n\n')
print(data.info(), '\n\n')

# Remove the first column since it's just indexing and it may confuse the ML algorithm
data.drop('Id', axis=1, inplace=True)
# checking for unknown numerical values (zeros)
print(data[(data['SepalLengthCm'] == 0) | (data['SepalWidthCm'] == 0) | (data['PetalLengthCm'] == 0) |
           (data['PetalWidthCm'] == 0)], '\n\n')
# Since there is no zero values in the dataset we didn't use the drop method to remove all the
# unknown numerical values (zeros)

print(data.head(), '\n\n')

# Preparing the dataset
# X = features values, all the columns except the last column
X = data.drop('Species', axis=1)
# y = the last column from the dataset (target values)
y = data['Species']

# Plotting features versus species
plt.xlabel('Features')
plt.ylabel('Species')

pltX = data.loc[:, 'SepalLengthCm']
pltY = data.loc[:, 'Species']
plt.scatter(pltX, pltY, color='blue', label='Sepal Length',
            alpha=0.3, edgecolors='none')

pltX = data.loc[:, 'SepalWidthCm']
pltY = data.loc[:, 'Species']
plt.scatter(pltX, pltY, color='green', label='Sepal Width',
            alpha=0.3, edgecolors='none')

pltX = data.loc[:, 'PetalLengthCm']
pltY = data.loc[:, 'Species']
plt.scatter(pltX, pltY, color='red', label='Petal Length',
            alpha=0.3, edgecolors='none')

pltX = data.loc[:, 'PetalWidthCm']
pltY = data.loc[:, 'Species']
plt.scatter(pltX, pltY, color='yellow', label='Petal Width',
            alpha=0.3, edgecolors='none')

plt.legend()
# plt.show()

# We need to encode our text values, as our model requires integer or float values, not string values .
# Applying Label Encoding on the target column dataset which is 'Species' .
# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'Species'.
data['Species'] = label_encoder.fit_transform(data['Species'])

print(data['Species'].unique(), '\n\n')  # Species names after labeling them

X = data.drop('Species', axis=1)
y = data['Species']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Training the model using DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Testing the model
predictions = model.predict(X_test)

# Evaluating the performance of our classifier
confusion_matrix(y_test, predictions)

# Checking the precision, recall, f1_score
print(classification_report(y_test, predictions), '\n\n')

# Training the model using RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions), '\n\n')

# Training the model using LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))

""" 
    Using DecisionTreeClassifier gave us a higher accuracy and values score than RandomForestClassifier 
    but using LogisticRegression model gave us a 100% accuracy and best score for the values which is 1.00,
    from the results we discovered that LogisticRegression is the best model to classify Iris-Species dataset .
"""

# End of code
