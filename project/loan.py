import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


data=pd.read_csv('project/loan_approval_data (2).csv')
# print(data.head())
# print(data.isnull().sum())

# that is used to devise the categorical and n umerical value are columns
catorical_cols=data.select_dtypes(include =['object']).columns
numerical_cols=data.select_dtypes(include =['float64','int64']).columns
# print(catorical_cols)
# print(numerical_cols)

# that is used to fill the null value in categorical and numerical columns
# imputer


num_imputer=SimpleImputer(strategy='mean')

data[numerical_cols]=num_imputer.fit_transform(data[numerical_cols])
# print(data.isnull().sum())
# 
cat_imputer=SimpleImputer(strategy='most_frequent')
data[catorical_cols]=cat_imputer.fit_transform(data[catorical_cols])
# print(data.isnull().sum())

# that is used to visualize the data and analyze the data

# classes_count=data['Loan_Approved'].value_counts()
# plt.pie(classes_count, labels=['no','yes'], autopct='%1.1f%%')
# plt.title('Is Loan Approved or Not')
# plt.show()

# that for gendar value



# gender_count = data['Gender'].value_counts()
# ax=sns.barplot(x=gender_count.index, y=gender_count.values)
# plt.title('Gender Distribution')
# ax.bar_label(ax.containers[0])
# plt.show()


# education_count = data['Education_Level'].value_counts()
# ax=sns.barplot(education_count)
# plt.title('Education Distribution')
# ax.bar_label(ax.containers[0])
# plt.show()

# sns.histplot(data=data,
#              x='Applicant_Income',
#              bins=20)

# plt.show()


# sns.histplot(data=data,
#              x='Coapplicant_Income',
#              bins=20)

# plt.show()






# sns.boxplot(
# data=data,
#  x='Loan_Approved',
#  y='Applicant_Income'
# )

# plt.show()


# fig, axes = plt.subplots(2,2)
# sns.boxplot(ax=axes[0,0], data=data, x='Loan_Approved', y='Applicant_Income')
# sns.boxplot(ax=axes[0,1], data=data, x='Loan_Approved', y='Coapplicant_Income')
# sns.boxplot(ax=axes[1,0], data=data, x='Loan_Approved', y='DTI_Ratio')
# sns.boxplot(ax=axes[1,1], data=data, x='Loan_Approved', y='Savings')

# plt.tight_layout()
# plt.show()


# sns.histplot(data=data,
#              x='Credit_Score',
#              hue='Loan_Approved',
#              bins=20,
#              multiple='dodge',)
# plt.show()



# sns.histplot(data=data,
#              x='Credit_Score',
#              hue='Applicant_Income',
#              bins=20,
#              multiple='dodge',)
# plt.show()