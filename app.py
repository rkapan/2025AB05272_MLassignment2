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
st.markdown("""
    <style>
    /* Target the metric containers */
    [data-testid="stMetric"] {
        background-color: #f0f2f6; /* Light grey background */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d1d1d1;
    }

    /* Target the label (the title like 'Best Model') */
    [data-testid="stMetricLabel"] {
        color: #1f77b4 !important; /* Blue color */
        font-weight: bold !important;
    }

    /* Target the value (the actual number/result) */
    [data-testid="stMetricValue"] {
        color: #d33682 !important; /* Magenta/Pink color */
    }

    h1, h2, h3 {
        color: #1f77b4 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    h1 {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }

    h2 {
        margin-top: 2rem !important;
        color: #d33682 !important; /* Matches your metric values */
    }

    /* Style the subheaders used for "Details per Model" */
    .stMarkdown h4 {
        color: #464b5d !important;
        background-color: #f0f2f6;
        padding: 5px 10px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }

    /* Style regular text/paragraphs */
    .stMarkdown p {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #31333F;
    }

    /* Style for the 'Details per Model' container labels specifically */
    .model-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%);
        border-top: 4px solid #d33682; /* Pink/Magenta accent */
        color: #ffffff !important;
        padding: 15px 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        font-family: 'Inter', sans-serif;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .model-header h3 {
        margin: 0 !important;
        font-size: 22px !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .section-header {
        color: #ffffff;
        background-color: #1f77b4;
        padding: 10px 15px;
        border-radius: 5px 5px 0 0;
        margin-top: 20px;
        font-weight: bold;
        display: block;
    }

    /* Add a border to the dataframe area to match the header */
    .report-container {
        border: 1px solid #464b5d; /* Darker border for dark theme */
        border-radius: 0 0 5px 5px;
        padding: 10px;
        background-color: transparent; /* Makes it blend in */
        margin-top: -10px; /* Pulls it up to connect with the header */
    }

    .main-section-header {
        color: #ffffff;
        background-color: #d33682; /* Pinkish-magenta to differentiate from the blue table header */
        padding: 12px 20px;
        border-radius: 8px;
        margin-top: 30px;
        margin-bottom: 20px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Header 4 - Plot/Table Titles (New) */
    .plot-header {
        font-size: 18px !important;
        font-weight: 500 !important;
        color: #e1e1e1 !important;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 5px 12px;
        border-radius: 4px;
        border-left: 3px solid #d33682; /* Pink accent to match metrics */
        margin-top: 15px;
        margin-bottom: 10px;
    }

    CSS
    /* Container for the uploader with a subtle border */
    .upload-box {
        border: 1px solid #d1d1d1;
        padding: 10px 15px;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        margin-bottom: 15px;
        max-width: 400px; /* Limits width for a cleaner look */
    }

    /* Small left-aligned text */
    .upload-label {
        text-align: left;
        font-size: 0.85rem !important;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 5px;
    }

    /* Target the Streamlit uploader to fit inside our border */
    [data-testid="stFileUploader"] {
        padding: 0;
    }

    CSS
    /* Right align and shrink the data source text */
    .right-aligned-text {
        text-align: right;
        font-size: 0.85rem !important;
        color: #1f77b4;
        margin-bottom: 5px;
        font-weight: 600;
    }

    /* Target the download button container to float right */
    div[data-testid="stColumn"] > div > div > div[data-testid="stVerticalBlock"] > div:has(button[kind="secondary"]) {
        display: flex;
        justify-content: flex-end;
    }

    /* Make the button itself more compact */
    button[kind="secondary"] {
        padding: 0.25rem 0.75rem !important;
        height: auto !important;
        font-size: 0.8rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
#st.title("2025AB05272 ML Model Evaluation App")
with st.sidebar:
    st.image("https://www.bits-pilani.ac.in/wp-content/uploads/bits-pillani-2-1.webp", width=100)
    st.title("Control Panel")
    
    
    model_dict = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }
    st.markdown("### 2. Model Configuration")
    model_names = st.multiselect(
        "Select Model(s) to Evaluate", 
        list(model_dict.keys()), 
        default=list(model_dict.keys())[:2]
    )
    
    st.info("App for ML Assignment-2 submission.")
col1, col2 = st.columns(2)
with col2:
    # Split col2 into 3 parts: 2 parts spacer, 1 part content
    _, _, download_col = st.columns([1, 1, 1.5])
    
    with download_col:
        st.markdown('<p class="right-aligned-text">📁 Data Source</p>', unsafe_allow_html=True)
        
        test_file_path = "data/initial/bank.csv"
        if os.path.exists(test_file_path):
            with open(test_file_path, "rb") as file:
                st.download_button(
                    label="📥 Download CSV", # Shortened label for better sizing
                    data=file,
                    file_name="test_data.csv",
                    mime="text/csv"
                )
    #st.subheader("Select Models to Evaluate")
    # model_dict = {
    #     "Logistic Regression": "logistic_regression",
    #     "Decision Tree": "decision_tree",
    #     "KNN": "knn",
    #     "Naive Bayes": "naive_bayes",
    #     "Random Forest": "random_forest",
    #     "XGBoost": "xgboost"
    # }

    # Allow multiple model selection
    #model_names = st.multiselect("Select Model(s)", list(model_dict.keys()), default=list(model_dict.keys())[:1])

with col1:
    # Wrap in a div to apply the border and alignment
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    
    # Left-aligned small text with icon
    st.markdown('<p class="upload-label">📤 Upload test data (CSV)</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.7rem; color: gray; margin-bottom: 10px;">Columns must match training data</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "hidden label", 
        type="csv", 
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and model_names:
    
    uploaded_file.seek(0)
    try:
        test_df = pd.read_csv(uploaded_file, sep=';')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop() # Stops execution if file is bad

    #st.write("Test data columns:", test_df.columns.tolist())

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
        st.subheader("🏆 Model Highlights")
        metrics_df = pd.DataFrame(metrics_list).set_index('Model')
        top_model = metrics_df.sort_values(by='Accuracy', ascending=False).index[0]
        best_acc = metrics_df.loc[top_model, 'Accuracy']
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Best Model", top_model)
        m2.metric("Highest Accuracy", f"{best_acc:.2%}")
        
        m3.metric("Best MCC", f"{metrics_df['MCC'].max():.2f}")

        st.divider()
        # Show metrics comparison table
        st.markdown('<span class="section-header">📊 Evaluation Metrics Comparison</span>', unsafe_allow_html=True)

        # Wrap the dataframe in the container WITHOUT the extra empty div
        with st.container():
            # We apply the styling directly to the dataframe display
            st.dataframe(
                metrics_df.style.highlight_max(axis=0, color='#0e4d25') # Darker green for dark mode
                                .highlight_min(axis=0, color='#4d0e0e') # Darker red for dark mode
                                .format(precision=3),
                use_container_width=True
            )

        # Show confusion matrices
        st.markdown('<div class="main-section-header">🔍 Details per Model: Deep Dive</div>', unsafe_allow_html=True)
        model_cm_items = list(confusion_matrices.items())
        n_models = len(model_cm_items)
        cols = 2
        rows = math.ceil(n_models / cols) if cols > 0 else 1
        grid = [st.columns(cols) for _ in range(rows)]
        ##rows = math.ceil(n_models / cols) if cols > 0 else 1
        for idx, (model_name, cm) in enumerate(model_cm_items):
            row = idx // cols
            col = idx % cols
            if row < len(grid) and col < len(grid[row]):
                with grid[row][col]:
                    # Use the border=True container to act as a 'Card'
                    with st.container(border=True):
                        # Styled Model Title
                        #st.markdown(f'<p class="model-header">{model_name}</p>', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="model-header">
                                {model_name}
                            </div>
                        """, unsafe_allow_html=True)
                        # --- Confusion Matrix Section ---
                        st.markdown('<p class="plot-header">📊 Confusion Matrix</p>', unsafe_allow_html=True)
                        st.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
                        plt.close('all')
                        fig, ax = plt.subplots(figsize=(6, 4))
                        # Using a slightly different colormap (e.g., 'Purples' or 'GnBu') for variety
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                            xticklabels=["Pred 0", "Pred 1"], 
                            yticklabels=["True 0", "True 1"], 
                            cbar=False, annot_kws={"size": 14})
                        fig.tight_layout()
                        st.pyplot(fig, width='stretch')
                        plt.close(fig)
                        
                        # --- Classification Report Section ---
                        st.markdown('<p class="plot-header">📋 Classification Report</p>', unsafe_allow_html=True)
                        report_df = pd.DataFrame(classification_report(y_true, predictions[model_name], output_dict=True)).transpose()
                        st.dataframe(report_df.style.format(precision=2), width='content')
        # model_cm_items = list(confusion_matrices.items())
        # n_models = len(model_cm_items)
        # cols = 2
        # rows = math.ceil(n_models / cols) if cols > 0 else 1

        # grid = [st.columns(cols) for _ in range(rows)]

        # for idx, (model_name, cm) in enumerate(model_cm_items):
        #     row = idx // cols
        #     col = idx % cols
        #     if row < len(grid) and col < len(grid[row]):
        #         with grid[row][col]:
        #             with st.container(border=True):
        #                 st.markdown(f"""
        #                     <div style="
        #                         border: 1px solid #d3d3d3; 
        #                         text-align: left; 
        #                         background-color: rgba(150, 150, 150, 0.05);
        #                         margin-bottom: 15px;">
        #                         <h3 style="margin: 0; color: gray;">{model_name}</h3>
        #                     </div>
        #                 """, unsafe_allow_html=True)
        #                 plt.clf()  
        #                 # Set figsize slightly larger (e.g., 7x5) to ensure internal text is readable
        #                 fig, ax = plt.subplots(figsize=(7, 5)) 
                        
        #                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
        #                             xticklabels=["Pred 0", "Pred 1"], 
        #                             yticklabels=["True 0", "True 1"], 
        #                             cbar=False) # Removing colorbar helps focus on the matrix itself
                        
        #                 # 3. Fine-tune layout to prevent clipping
        #                 plt.tight_layout()

        #                 # 4. Render with forced container width for "Proper Size"
        #                 st.pyplot(fig, use_container_width=True)
        #                 plt.close(fig)
                        
        #                 st.markdown(f"""
        #                     <div style="border-bottom: 2px solid gray; width: fit-content; margin-bottom: 10px;">
        #                         <h4 style="color: gray; margin: 0; padding-bottom: 5px;">Confusion Matrix</h4>
        #                     </div>
        #                 """, unsafe_allow_html=True)
        #                 st.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
                        
        #                 st.markdown(f"""
        #                     <div style="border-bottom: 2px solid gray; width: fit-content; margin-bottom: 10px;">
        #                         <h4 style="color: gray; margin: 0; padding-bottom: 5px;">Classification Report</h4>
        #                     </div>
        #                 """, unsafe_allow_html=True)
        #                 #st.text(classification_report(y_true, predictions[model_name]))
        #                 report_df = pd.DataFrame(classification_report(y_true, predictions[model_name], output_dict=True)).transpose()
        #                 st.dataframe(report_df.style.format(precision=2), use_container_width=True)

                     




        # Optionally, fill remaining cells in the last row to keep the grid neat
        last_row = len(grid) - 1
        for col in range(len(model_cm_items) % cols, cols):
            if last_row >= 0 and col < len(grid[last_row]):
                with grid[last_row][col]:
                    st.write("")
        # for idx, (model_name, cm) in enumerate(model_cm_items):
        #     row = idx // cols
        #     col = idx % cols
        #     if row < len(grid) and col < len(grid[row]):
        #         with grid[row][col]:
        #             with st.container(border=True, height=850):
        #                 st.markdown(f"{model_name}")
        #                 plt.clf()  # Clear the current figure to avoid overlap
        #                 fig, ax = plt.subplots(figsize=(6, 5))
        #                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
        #                             xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"],cbar=False)
        #                 ax.set_title(f"Confusion Matrix - {model_name}")
        #                 plt.tight_layout()
        #                 st.pyplot(fig,use_container_width=True)
        #                 plt.close(fig)

        # # Optionally, fill remaining cells in the last row to keep the grid neat
        # last_row = len(grid) - 1
        # for col in range(len(model_cm_items) % cols, cols):
        #     if last_row >= 0 and col < len(grid[last_row]):
        #         with grid[last_row][col]:
        #             st.write("")
        # Show classification reports
        # st.subheader("Classification Reports")
        # for model_name in model_names:
        #     if model_name in predictions:
        #         st.write(f"**{model_name}**")
        #         st.text(classification_report(y_true, predictions[model_name]))