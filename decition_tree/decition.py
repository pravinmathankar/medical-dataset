import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('decition_tree/tested.csv')

# Fill missing numerical values
feature = ['Age', 'Fare']

imputer = SimpleImputer(strategy='mean')
data[feature] = imputer.fit_transform(data[feature])

# Fill missing categorical values
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Encode categorical columns
label_encoder = LabelEncoder()

data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

# Select only useful columns
x = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = data['Survived']

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()

model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)