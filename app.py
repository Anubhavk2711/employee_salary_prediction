import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Income Prediction Web App")
st.write("This app predicts whether a person's income is >50K or <=50K")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Raw Data", df.head())

    # Handle missing values
    df.replace(" ?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical features
    df_clean = df.copy()
    label_encoders = {}
    for col in df_clean.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le

    # Features and target
    X = df_clean.drop('income', axis=1)
    y = df_clean['income']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # User input
    st.write("### Enter New User Data")
    user_input = {}
    for col in X.columns:
        if df[col].dtype == 'object':
            options = df[col].dropna().unique().tolist()
            user_input[col] = st.selectbox(col, options)
        else:
            user_input[col] = st.number_input(col, value=float(df[col].median()))

    # Preprocess user input
    input_df = pd.DataFrame([user_input])
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    label = label_encoders['income'].inverse_transform([prediction])[0]

    st.write("### 🧠 Prediction")
    st.success(f"Predicted Income: {label}")
