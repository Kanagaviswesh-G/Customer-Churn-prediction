import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset (assuming it's saved as a CSV file)
data = pd.read_csv('C:\\Users\\viswe\\OneDrive\\Desktop\\Customer churn prediction\\Telco-Customer-Churn.csv')


# Fill missing values in 'TotalCharges' with 0 (or you could use the median/mean)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.fillna(0, inplace=True)

# Encoding categorical columns using Label Encoding
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']  # Exclude 'Churn' from this list

# Apply LabelEncoder to categorical columns (excluding 'Churn')
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Ensure conversion to string before encoding

# Convert 'Churn' to numerical values (0 and 1)
data['Churn'] = data['Churn'].replace({'No': 0, 'Yes': 1})

# Ensure that 'Churn' is an integer
data['Churn'] = pd.to_numeric(data['Churn'], errors='coerce')

# Features (X) and Target (y)
X = data.drop(['customerID', 'Churn'], axis=1)  # Drop customerID and target column
y = data['Churn']  # Churn is the target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing numerical features (e.g., MonthlyCharges, Tenure, TotalCharges)
scaler = StandardScaler()

# Apply scaling only to numeric columns
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train= scaler.fit_transform(X_train[numeric_columns])
X_test = scaler.transform(X_test[numeric_columns])

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Convert numerical predictions (0, 1) back to categorical values ('No', 'Yes')
y_pred_categorical = ['Yes' if x == 1 else 'No' for x in y_pred]

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Optionally, print a few of the predictions along with the actual values
predictions_df = pd.DataFrame({
    'Actual': ['Yes' if x == 1 else 'No' for x in y_test],
    'Predicted': y_pred_categorical
})

print(predictions_df.head())  # Display the first few actual vs predicted values
