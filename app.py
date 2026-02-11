import math
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model.datapreprocessing import fetch_processed_data_from_df
from model.evaluate import fetch_metrics
from sklearn.metrics import confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide") 
st.title("2025AB05272 ML Model Evaluation App")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Select Models to Evaluate")
    model_dict = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }

    # Allow multiple model selection
    model_names = st.multiselect("Select Model(s)", list(model_dict.keys()), default=list(model_dict.keys())[:1])

with col2:
    st.write("Upload test data (CSV, columns must match training data):")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and model_names:
    
    uploaded_file.seek(0)
    try:
        test_df = pd.read_csv(uploaded_file, sep=';')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop() # Stops execution if file is bad

    st.write("Test data columns:", test_df.columns.tolist())

    metrics_list = []
    confusion_matrices = {}
    predictions = {}

    # 2. LOAD SCALER ONCE OUTSIDE THE LOOP (Unless you have different scalers per model)
    scaler_path = os.path.join("model/pkl", "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        # Pre-process the data once
        X_scaled, y = fetch_processed_data_from_df(test_df, scaler)
    else:
        st.error(f"Scaler file not found: {scaler_path}")
        st.stop()

    

    for model_name in model_names:
        model_file_key = model_dict[model_name]
        model_path = os.path.join("model/pkl", f"{model_file_key}.pkl")
        scaler_path=os.path.join("model/pkl", f"scaler.pkl")
        print("------------------------------------------------------")
        print(scaler_path)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.error(f"Model file not found: {model_path}")
            continue
        
        #test_df = pd.read_csv(uploaded_file, sep=';')
        
        X_scaled, y = fetch_processed_data_from_df(test_df,scaler)
        #st.write("Test data columns:", X.columns.tolist())
        y_true = y if y is not None else None
        y_pred = model.predict(X_scaled)
        predictions[model_name] = y_pred

        if y_true is not None:
            metrics = fetch_metrics(model, X_scaled, y_true)
            metrics['Model'] = model_name
            metrics_list.append(metrics)
            cm = confusion_matrix(y_true, y_pred)
            confusion_matrices[model_name] = cm
        else:
            st.subheader(f"Predictions for {model_name}")
            st.write(y_pred)

    if y_true is not None and metrics_list:
        # Show metrics comparison table
        st.subheader("Evaluation Metrics Comparison")
        metrics_df = pd.DataFrame(metrics_list).set_index('Model')
        st.dataframe(metrics_df)

        # Show confusion matrices
        st.markdown(f"## Details per Model")
        model_cm_items = list(confusion_matrices.items())
        n_models = len(model_cm_items)
        cols = min(3, n_models) if n_models > 0 else 1
        rows = math.ceil(n_models / cols) if cols > 0 else 1

        grid = [st.columns(cols) for _ in range(rows)]

        for idx, (model_name, cm) in enumerate(model_cm_items):
            row = idx // cols
            col = idx % cols
            if row < len(grid) and col < len(grid[row]):
                with grid[row][col]:
                    with st.container(border=True):
                        st.markdown(f"""
                            <div style="
                                border: 1px solid #d3d3d3; 
                                text-align: left; 
                                background-color: rgba(150, 150, 150, 0.05);
                                margin-bottom: 15px;">
                                <h3 style="margin: 0; color: gray;">{model_name}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style="border-bottom: 2px solid gray; width: fit-content; margin-bottom: 10px;">
                                <h4 style="color: gray; margin: 0; padding-bottom: 5px;">Confusion Matrix</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        st.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
                        ax.set_title(f"Confusion Matrix - {model_name}")
                        st.pyplot(fig)

                        st.markdown(f"""
                            <div style="border-bottom: 2px solid gray; width: fit-content; margin-bottom: 10px;">
                                <h4 style="color: gray; margin: 0; padding-bottom: 5px;">Classification Report</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        #st.text(classification_report(y_true, predictions[model_name]))
                        report_df = pd.DataFrame(classification_report(y_true, predictions[model_name], output_dict=True)).transpose()
                        st.dataframe(report_df.style.format(precision=2), use_container_width=True)

        # Optionally, fill remaining cells in the last row to keep the grid neat
        last_row = len(grid) - 1
        for col in range(len(model_cm_items) % cols, cols):
            if last_row >= 0 and col < len(grid[last_row]):
                with grid[last_row][col]:
                    st.write("")

        # Show classification reports
        # st.subheader("Classification Reports")
        # for model_name in model_names:
        #     if model_name in predictions:
        #         st.write(f"**{model_name}**")
        #         st.text(classification_report(y_true, predictions[model_name]))