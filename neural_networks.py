import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Page Configuration
st.set_page_config(page_title="Neural Networks", layout="wide")

def run():
    st.markdown("""
        <style>
            .stButton > button {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.5em 2em;
                font-size: 18px;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #c0392b;
            }
            .block {
                background-color: #f9f9f9;
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Neural Network Classification Dashboard")

    if st.button("⬅️ Back to Services"):
        st.session_state.page = "services"
        st.rerun()

    st.markdown("---")

    # Upload dataset
    with st.container():
        with st.expander("📤 Upload Dataset", expanded=True):
            uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file).dropna()
        st.markdown("### 🧾 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Select target
        target_col = st.selectbox("🎯 Select Target Column", df.columns)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        # Define PyTorch model
        class NeuralNet(nn.Module):
            def __init__(self, input_dim, hidden1=64, hidden2=32, output_dim=2):
                super(NeuralNet, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden1),
                    nn.ReLU(),
                    nn.Linear(hidden1, hidden2),
                    nn.ReLU(),
                    nn.Linear(hidden2, output_dim)
                )

            def forward(self, x):
                return self.net(x)

        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train))
        model = NeuralNet(input_dim, output_dim=output_dim)

        st.markdown("### ⚙️ Hyperparameters")
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Number of Epochs", 1, 100, 10)
        with col2:
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        with st.spinner("Training model..."):
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

        st.success("✅ Training Complete!")

        with torch.no_grad():
            train_preds = torch.argmax(model(X_train_tensor), dim=1).numpy()
            val_preds = torch.argmax(model(X_val_tensor), dim=1).numpy()

        train_accuracy = accuracy_score(y_train, train_preds)
        val_accuracy = accuracy_score(y_val, val_preds)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].bar(['Train', 'Validation'], [train_accuracy, val_accuracy], color=['green', 'blue'])
        axs[0].set_title("Accuracy")

        axs[1].bar(['Train', 'Validation'], [1 - train_accuracy, 1 - val_accuracy], color=['green', 'blue'])
        axs[1].set_title("Loss")

        st.pyplot(fig)

        # Upload test data
        st.markdown("### 🤖 Make Predictions on New Data")
        uploaded_test_file = st.file_uploader("📄 Upload CSV for Predictions", type=["csv"])

        if uploaded_test_file:
            test_df = pd.read_csv(uploaded_test_file)
            st.write("📋 Test Sample Preview")
            st.dataframe(test_df.head(), use_container_width=True)

            X_test = test_df.drop(columns=[target_col], errors='ignore')
            X_test = pd.DataFrame(X_test, columns=X_train.columns)
            X_test_scaled = scaler.transform(X_test)

            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            with torch.no_grad():
                predictions = torch.argmax(model(X_test_tensor), dim=1).numpy()

            test_df['Prediction'] = predictions
            st.markdown("### 🔍 Prediction Results")
            st.dataframe(test_df, use_container_width=True)

            csv = test_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv, file_name="predictions.csv", mime="text/csv")

        # Confusion Matrix
        st.markdown("### 📊 Confusion Matrix")
        cm = confusion_matrix(y_val, val_preds)
        fig_cm = plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
                    xticklabels=le.classes_ if y.dtype == 'object' else ["0", "1"], 
                    yticklabels=le.classes_ if y.dtype == 'object' else ["0", "1"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig_cm)

    else:
        st.info("Please upload a dataset to start.")
