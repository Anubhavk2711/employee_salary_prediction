import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Employee Salary Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")  # 🔁 REPLACE WITH YOUR FILE PATH
    return df

data = load_data()

# Encode categorical columns
def preprocess(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

# Preprocessing
df_clean = preprocess(data)

# Split into features and target
X = df_clean.drop('salary', axis=1)  # 🔁 Change 'salary' if your column name differs
y = df_clean['salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model inside app
model = RandomForestClassifier()
model.fit(X_train, y_train)

# User input section
st.sidebar.header("Input Employee Features")

input_data = {}
for col in X.columns:
    if df_clean[col].nunique() <= 10:
        val = st.sidebar.selectbox(f"{col}", df_clean[col].unique())
    else:
        val = st.sidebar.number_input(f"{col}", float(df_clean[col].min()), float(df_clean[col].max()), float(df_clean[col].mean()))
    input_data[col] = val

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict Salary Category"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Salary Category: {prediction}")
