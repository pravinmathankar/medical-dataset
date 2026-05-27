import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('logistic_regration/1-heart.csv')
# print(data.head())

x=data.drop(columns=['target'])
y=data['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
knn_model= KNeighborsClassifier(n_neighbors=7pwd)
knn_model.fit(x_train_scaled,y_train)
y_pred = knn_model.predict(x_test_scaled)

print("recall:", recall_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))    
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("accuracy:", accuracy_score(y_test, y_pred))