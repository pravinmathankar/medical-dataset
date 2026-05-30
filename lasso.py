# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from pyparsing import alphas
# from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import Lasso
# # from sklearn.linear_model import lassoCV
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

# data = pd.read_csv('insurance.csv')

# x = data.drop(columns=['charges'])
# y = data['charges']

# # that is a categorical variable, we need to encode it
# # and that for is a one-hot encoding, we can use pandas get_dummies function for that 
# # he seprate the region column into 4 columns, and each column will have 0 or 1 depending on the value of the region column
# x = pd.get_dummies(columns=['region'], data=x, drop_first=True, dtype=int)
# x['sex'] = x['sex'].map({'male': 0, 'female': 1})
# x['smoker'] = x['smoker'].map({'yes': 0, 'no': 1})


# # thats a good idea to create interaction features, because it can help the model to capture the non-linear relationships between the features and the target variable
# # and that is a good idea to create interaction features, because it can help the model to capture the non-linear relationships between the features and the target variable
# x['age_smoker'] = x['age'] * x['smoker']
# x['bmi_smoker'] = x['bmi'] * x['smoker']


# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # print(x.head(5))


# # sns.heatmap(x.corr(), annot=True, cmap='coolwarm')
# # plt.show()

# # lasso_model = Lasso(alpha=0.1)
# # lasso_model.fit(x_train, y_train)
# # y_pred = lasso_model.predict(x_test)

# # alphas = [0.001, 0.1, 1, 2, 5, 10, 20, 30, 40, 50, 100]
# # mses = []

# # for a in alphas:
# #     lasso_model = Lasso(alpha=a)
# #     lasso_model.fit(x_train, y_train)

# #     y_pred = lasso_model.predict(x_test)
# #     mse = mean_squared_error(y_test, y_pred)
# #     print(f"MSE for alpha={a}: ", mse)
# #     mses.append(mse)
    
# # sns.lineplot(x=alphas, y=mses, marker="o")

# from sklearn.linear_model import LassoCV
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# a = [0.001, 0.1, 1, 2, 5, 10, 20, 30, 40, 50, 100]

# lasso_cv_model = LassoCV(
#     alphas=a,
#     cv=5,
#     max_iter=1000,
#     random_state=42
# )

# lasso_cv_model.fit(x_train, y_train)

# print("best alpha: ", lasso_cv_model.alpha_)

# y_pred = lasso_cv_model.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
# print("Adjusted R2: ", adjusted_r2)
# print("mse = ", mse)
# print("s2 = ", r2)

import pandas as pd

data = pd.DataFrame({
    'City':['Delhi','Mumbai','Bhopal']
})

encoded = pd.get_dummies(data,columns=['City'],prefix='city_name', dtype=int)

print(encoded)