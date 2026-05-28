import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB






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




# that is uded to encoding the data
# one hot incoder :=  that for userd to ordinal values ek hee column me hota hai   



# label incoder :=  that is used to nominal data  means




le = LabelEncoder()
data["Education_Level"] = le.fit_transform(data["Education_Level"])
data["Loan_Approved"] = le.fit_transform(data["Loan_Approved"])

# print(data.head())
cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(data[cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=data.index)
data = pd.concat([data.drop(columns=cols), encoded_df], axis=1)
# print(df.head())
# print(data.info())


nums=data.select_dtypes(include=['float64','int64']).columns
corelation_matrix=data[nums].corr()
# print(corelation_matrix)



sns.heatmap(corelation_matrix, annot=True, cmap='coolwarm',fmt='.2f')
plt.title('Correlation Matrix')
# plt.show()


x=data.drop(columns=['Loan_Approved'])
y=data['Loan_Approved']

x_train,x_test,y_train,y_test=train_test_split(data.drop(columns=['Loan_Approved']),data['Loan_Approved'],test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


# print(x_train_scaled)
# print(x_test_scaled)

# logreg=LogisticRegression()
# logreg.fit(x_train_scaled,y_train)

# logreg_pred=logreg.predict(x_test_scaled)

# print("Classification Report:\n", classification_report(y_test, logreg_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, logreg_pred))
# print("Accuracy Score:", accuracy_score(y_test, logreg_pred))
# print("Accuracy Score:", accuracy_score(y_test, logreg_pred))
# print("Precision Score:", precision_score(y_test, logreg_pred))
# print("Recall Score:", recall_score(y_test, logreg_pred))
# print("F1 Score:", f1_score(y_test, logreg_pred))



# knnnn


# knn=KNeighborsClassifier(n_neighbors=7)
# knn.fit(x_train_scaled,y_train)
# knn_pred=knn.predict(x_test_scaled)

# print("Classification Report:\n", classification_report(y_test, knn_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))    
# print("Accuracy Score:", accuracy_score(y_test, knn_pred))
# print("Precision Score:", precision_score(y_test, knn_pred))
# print("Recall Score:", recall_score(y_test, knn_pred))
# print("F1 Score:", f1_score(y_test, knn_pred))




# that is navi byes algorithm

nb=GaussianNB()
nb.fit(x_train_scaled,y_train)
y_pred=nb.predict(x_test_scaled)


print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Precision Score:", precision_score(y_test, y_pred))
print("Recall Score:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))










data['DTI_Ratio_sq'] = data['DTI_Ratio']**2
data['Credit_score'] = data['Credit_Score']**2

data['Application_Income_log'] = np.log1p(data['Applicant_Income'])
x=data.drop(columns=['Loan_Approved','DTI_Ratio','Credit_Score','Applicant_Income'])
y=data['Loan_Approved']