import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the dataset (same steps as your original code)
data = pd.read_csv('C:\\Users\\viswe\\OneDrive\\Desktop\\Customer churn prediction\\Telco-Customer-Churn.csv')

# Fill missing values in 'TotalCharges' with 0 (or you could use the median/mean)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.fillna(0, inplace=True)

# Encoding categorical columns using Label Encoding
categorical_columns = ['gender', 'PaymentMethod']

# Apply LabelEncoder to categorical columns (excluding 'Churn')
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Ensure conversion to string before encoding

# Convert 'Churn' to numerical values (0 and 1)
data['Churn'] = data['Churn'].replace({'No': 0, 'Yes': 1})

# Features (X) and Target (y)
X = data[['gender', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]  # Only the required columns
y = data['Churn']  # Churn is the target variable

# Standardizing numerical features (MonthlyCharges, TotalCharges)
scaler = StandardScaler()
X[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X[['MonthlyCharges', 'TotalCharges']])

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Function to preprocess user input and make a prediction
def predict_churn(customer_id, gender, payment_method, monthly_charges, total_charges):
    # Display the customer ID in the output for reference
    print(f"Customer ID: {customer_id}")

    # Create a dictionary from the input
    input_data = {
        'gender': [gender],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame(input_data)

    # Apply the same Label Encoding for categorical features
    for col in categorical_columns:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col].astype(str))

    # Standardize the numerical features in the input
    input_df[['MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['MonthlyCharges', 'TotalCharges']])

    # Predict the churn (0 or 1)
    prediction = model.predict(input_df)

    # Convert prediction to 'Yes' or 'No'
    return 'No' if prediction[0] == 1 else 'Yes'


# Create a Gradio Interface
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Textbox(label='Customer ID'),  # Added customerID as a textbox input
        gr.Radio(['Male', 'Female'], label='Gender'),
        gr.Radio(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label='PaymentMethod'),
        gr.Slider(minimum=0, maximum=200, step=0.1, label='MonthlyCharges'),
        gr.Slider(minimum=0, maximum=10000, step=0.1, label='TotalCharges'),
    ],
    outputs=gr.Textbox(label="Churn Prediction (Yes/No)"),
    live=True
)

# Launch the Gradio interface
interface.launch()
