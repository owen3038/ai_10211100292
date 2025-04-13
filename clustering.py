import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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

            .upload-box .css-1d391kg {
                border: 2px dashed #9B6BE8;
            }

            .data-preview {
                background-color: #F6F6FB;
                padding: 1em;
                border-radius: 10px;
                margin-bottom: 1em;
            }

            .cluster-section {
                background-color: #f1f4ff;
                padding: 1.5em;
                border-radius: 14px;
                box-shadow: 0 2px 8px rgba(90, 85, 174, 0.1);
                margin-top: 20px;
            }

            .download-button > button {
                margin-top: 20px;
                background-color: #5A55AE;
                border-radius: 8px;
                font-size: 16px;
                color: white;
            }

            .download-button > button:hover {
                background-color: #473B9B;
            }
        </style>
        <div class="main-title">🔄 K-Means Clustering Visualizer</div>
    """, unsafe_allow_html=True)

    # Back button
    if st.button("⬅️ Back to Services"):
        st.session_state.page = "services"
        st.rerun()

    # Upload CSV
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("🚫 The uploaded CSV is empty.")
                return

            st.subheader("📋 Dataset Preview")
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            st.write(df.head())
            st.markdown('</div>', unsafe_allow_html=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                st.error("🚫 No numeric columns found.")
                return

            selected_features = st.multiselect("🧮 Select 2 or 3 numeric columns for clustering:", numeric_cols)

            if len(selected_features) not in [2, 3]:
                st.warning("⚠️ Please select exactly 2 or 3 columns.")
            else:
                st.markdown('<div class="cluster-section">', unsafe_allow_html=True)

                X = df[selected_features]
                k = st.slider("🔢 Choose number of clusters (k):", 2, 10, 3)

                model = KMeans(n_clusters=k, random_state=42)
                df["Cluster"] = model.fit_predict(X)
                centers = model.cluster_centers_

                st.success("✅ Clustering completed successfully!")
                st.write("📍 Cluster Centers:")
                st.write(pd.DataFrame(centers, columns=selected_features))

                # Visualization
                st.subheader("📊 Cluster Plot")
                fig = plt.figure()

                if len(selected_features) == 2:
                    sns.scatterplot(x=selected_features[0], y=selected_features[1],
                                    hue="Cluster", data=df, palette="Set2", s=80)

                    plt.scatter(centers[:, 0], centers[:, 1],
                                c='black', s=200, marker='X', label='Centroids')
                    plt.legend()

                else:
                    from mpl_toolkits.mplot3d import Axes3D
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(df[selected_features[0]], df[selected_features[1]], df[selected_features[2]],
                               c=df["Cluster"], cmap='Set2', s=80)

                    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                               c='black', s=200, marker='X', label='Centroids')

                    ax.set_xlabel(selected_features[0])
                    ax.set_ylabel(selected_features[1])
                    ax.set_zlabel(selected_features[2])
                    ax.legend()

                st.pyplot(fig)

                # Download result
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Clustered Data",
                                   data=csv,
                                   file_name="clustered_data.csv",
                                   mime="text/csv",
                                   key="download_clustered",
                                   help="Download your dataset with cluster labels.")

                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error while processing: {e}")
    else:
        st.info("👈 Upload a CSV file to begin clustering.")
