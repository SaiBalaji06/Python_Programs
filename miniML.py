import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neighbors
import matplotlib.pyplot as plt

iris = pd.read_csv("C:\\Users\\saibalaji\\Desktop\\iris2.zip")
print(iris)
x = iris.drop(columns=['class'])
y = iris['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#dstree algorithm
print("Decision Tree Algorithm")
dstreeclassifier = tree.DecisionTreeClassifier()
dstreeclassifier.fit(x_train, y_train)
dstreepredictions = dstreeclassifier.predict(x_test)
print("Accuracy Score =", accuracy_score(y_test, dstreepredictions)*100)
#knn algorithm
print("K Neighbours Algorithm")
knclassifier = neighbors.KNeighborsClassifier()
knclassifier.fit(x_train, y_train)
knprediction = knclassifier.predict(x_test)
print("Accuracy score =", accuracy_score(y_test, knprediction)*100)
#comparison graph
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
names = ['KNN', 'Decision Tree']
result = [96.66, 93.33]
plt.bar(names, result)
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.show()
