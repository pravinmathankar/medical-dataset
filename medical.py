import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv('insurance.csv')


print(data.head())
print(data.columns)
# # visulization
sns.scatterplot(x=data["bmi"], y=data["charges"],hue=data["smoker"])

plt.show()

# x=data.drop(columns=['charges','region'])
# x=data.drop(columns=['charges'])
# y=data['charges']

# x['sex']=x['sex'].map({'male':0,'female':1})
# x['smoker']=x['smoker'].map({'yes':0,'no':1})

# x=pd.get_dummies(x['region'],drop_first=True ,dtype=int)


# print(x.head(5))

# print(x.head(5))


# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# model = LinearRegression()
# model.fit(x_train,y_train)

# print(x_train.head(5))
# y_pred = model.predict(x_test)
# print(y_pred)
# print(y_test)
# r_squared = r2_score(y_test, y_pred)
# print("R-squared:", r_squared)
# n=x_test.shape[0]
# p=x_test.shape[1]
# adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
# print("Adjusted R-squared:", adjusted_r_squared)



# x=data.drop(columns=['charges'])
# y=data['charges']

# x =pd.get_dummies(columns=['region'],data=x,drop_first=False,dtype=int)

# x['sex']=x['sex'].map({'male':0,'female':1})
# x['smoker']=x['smoker'].map({'yes':0,'no':1})

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# model = LinearRegression()
# model.fit(x_train,y_train)

# y_pred = model.predict(x_test)
# print(y_pred)



# # intraction feature

# x=data.drop(columns=['charges'])
# y=data['charges']

# x =pd.get_dummies(columns=['region'],data=x,drop_first=False,dtype=int)

# x['sex']=x['sex'].map({'male':0,'female':1})
# x['smoker']=x['smoker'].map({'yes':0,'no':1})
# x['agr_smoker']=x['age']*x['smoker']
# x['bmi_smoker']=x['bmi']*x['smoker']
# x['bmi_smoker']=x['age']*x['smoker']
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# print(x.head(5))


# model = LinearRegression()
# model.fit(x_train,y_train)

# y_pred = model.predict(x_test)

# r_squared = r2_score(y_test, y_pred)
# print("R-squared:", r_squared)
# n=x_test.shape[0]
# p=x_test.shape[1]
# adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
# print("Adjusted R-squared:", adjusted_r_squared)



# # check the model is overfitting or underfitting
# y_pred = model.predict(x_test)
# y_train_pred =model.predict(x_train)
# r2_train = r2_score(y_train, y_train_pred)
# print("R-squared for training data:", r2_train)
# r2= r2_score(y_test, y_pred)
# print("R-squared for test data:", r2)