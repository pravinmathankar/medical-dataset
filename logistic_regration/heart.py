import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score
from sklearn.preprocessing import StandardScaler
data=pd.read_csv('logistic_regration/1-heart.csv')
# print(data.head())
# print(data.isnull().sum())

x=data.drop('target',axis=1)
y=data['target']

# print(x.head())
# print(y.head())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=1000)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
# y_pred=model.predict(x_test)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)




print("Accuracy:",accuracy_score(y_test,y_pred))
# print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
# print("Classification Report:\n",classification_report(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))














# y=y_train[y_train==1]
# print(y)
# y=y_train[y_train==0]
# print(y)

