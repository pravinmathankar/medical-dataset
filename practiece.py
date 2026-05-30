import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


data=pd.read_csv('insurance.csv')
# print(data.head(5))


x=data.drop(columns=['charges'])
y=data['charges']

x['sex']=x['sex'].map({'male':0,'female':1})
x['smoker']=x['smoker'].map({'yes':0,'no':1})



x = pd.get_dummies(x, columns=['region'], drop_first=True, dtype=int)


x['agr_smoker']=x['age']*x['smoker']
x['bmi_smoker']=x['bmi']*x['smoker']
# print(x.head(5))

# print(y.head(5))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


print("accuracy:",accuracy_score(y_test,y_pred))

# r_squared = r2_score(y_test, y_pred)
# print("R-squared:", r_squared)
# adjeusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
# print("Adjusted R-squared:", adjeusted_r_squared)
# print(x.head(5))

# sns.scatterplot(x=data["bmi"], y=data["charges"],hue=data["smoker"])
# plt.show()