# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

# Data preprocessing function
def preprocess_data(data, is_test=False):
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    if 'Fare' in data.columns:  # For test data, Fare might be missing
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Convert categorical data to numeric
    data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
    data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
    
    # Feature engineering: Drop columns not useful for prediction
    drop_columns = ['Name', 'Ticket', 'Cabin']  # Keep PassengerId for test data
    if not is_test:  # Only drop PassengerId for training data
        drop_columns.append('PassengerId')
    data.drop(columns=drop_columns, inplace=True, errors='ignore')
    
    return data

# Preprocess train and test data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data, is_test=True)  # Preserve PassengerId

# Splitting data into features and target
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ensure 'PassengerId' is not used for scaling
test_data_features = test_data.drop(columns=['PassengerId'], errors='ignore')
test_data_scaled = scaler.transform(test_data_features)

# Splitting the data for validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm = SVC()
svm.fit(X_train, y_train)

# Predict on the test set
test_predictions = svm.predict(test_data_scaled)

# Save the predictions to a CSV file
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
output.to_csv('submission.csv', index=False)