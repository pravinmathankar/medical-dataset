import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

data =pd.read_csv('logistic_regration/1-heart.csv')
# print(data.head())

x = data.drop(columns=['target'])
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

y_pred = nb_model.predict(x_test)
# print(x_test.head())


print("recall:", recall_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("accuracy:", nb_model.score(x_test, y_test))
print("r2 score:", r2_score(y_test, y_pred))
