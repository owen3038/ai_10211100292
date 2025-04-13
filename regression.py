import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run():
    st.markdown("""
        <style>
            .main-title {
                font-size: 40px;
                text-align: center;
                color: #5A55AE;
                font-weight: 800;
                margin-bottom: 10px;
            }

            .stButton > button {
                background: linear-gradient(90deg, #5A55AE, #9B6BE8);
                color: white;
                font-weight: 600;
                font-size: 18px;
                padding: 0.6em 1.8em;
                border: none;
                border-radius: 12px;
                transition: 0.3s ease-in-out;
            }

            .stButton > button:hover {
                background: linear-gradient(90deg, #473B9B, #7B4DE4);
                transform: scale(1.03);
            }

            .data-preview, .card-section {
                background-color: #F6F6FB;
                padding: 1em;
                border-radius: 12px;
                margin-top: 15px;
                box-shadow: 0 2px 8px rgba(90, 85, 174, 0.1);
            }
        </style>
        <div class="main-title">📈 Regression Analysis Tool</div>
    """, unsafe_allow_html=True)

    if st.button("⬅️ Back to Services"):
        st.session_state.page = "services"
        st.rerun()

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("📋 Dataset Preview")
        st.markdown('<div class="data-preview">', unsafe_allow_html=True)
        st.write(data.head())
        st.markdown('</div>', unsafe_allow_html=True)

        if data.shape[1] < 2:
            st.warning("⚠️ Please upload a CSV with at least two columns.")
            return

        feature_columns = data.columns.tolist()
        target_column = st.selectbox("🎯 Select Target Column", options=feature_columns)

        feature_columns.remove(target_column)
        selected_features = st.multiselect("🧮 Select Feature Columns", options=feature_columns, default=feature_columns)

        if len(selected_features) > 0:
            st.markdown('<div class="card-section">', unsafe_allow_html=True)

            st.subheader("🛠️ Preprocessing Options")
            handle_missing = st.checkbox("🔧 Handle Missing Data", value=True)
            normalize_data = st.checkbox("📐 Normalize Numeric Data", value=False)

            X = data[selected_features]
            y = data[target_column]

            if handle_missing:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

            if normalize_data:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
            numeric_columns = X.select_dtypes(exclude=['object']).columns.tolist()

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(), categorical_columns),
                ('num', 'passthrough', numeric_columns)
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])

            if st.button("🚀 Run Regression"):
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.subheader("📊 Regression Results")
                st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
                st.write(f"**MSE (Mean Squared Error):** {mse:.2f}")
                st.write(f"**R² Score:** {r2:.2f}")

                st.subheader("🔮 Predicted vs Actual")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, color='blue')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Scatter Plot: Predicted vs Actual")
                st.pyplot(fig)

                if len(selected_features) == 1:
                    st.subheader("🧪 Custom Prediction")
                    val = st.number_input(f"Enter value for {selected_features[0]}:", min_value=0.0, value=0.0)
                    prediction = pipeline.predict([[val]])
                    st.write(f"📌 Predicted value: {prediction[0]}")

                    st.subheader("📉 Regression Line")
                    fig, ax = plt.subplots()
                    ax.scatter(X_test, y_test, color='blue', label='Actual')
                    ax.plot(X_test, pipeline.predict(X_test), color='red', label='Line')
                    ax.set_xlabel(selected_features[0])
                    ax.set_ylabel(target_column)
                    ax.set_title("Regression Line")
                    ax.legend()
                    st.pyplot(fig)

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("📎 Please upload a CSV file to begin regression.")
