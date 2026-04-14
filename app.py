import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# --- Page Config ---
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# --- Data Loading ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Pregnancies', 'Age', 'BloodPressure', 'SkinThickness', 'Insulin', 'Outcome']
    return pd.read_csv(url, names=cols)

# Initialize Session States
if 'original_df' not in st.session_state:
    st.session_state.original_df = load_data()
if 'df' not in st.session_state:
    st.session_state.df = st.session_state.original_df.copy()
if 'clean_msg' not in st.session_state:
    st.session_state.clean_msg = ""
if 'active_features' not in st.session_state:
    st.session_state.active_features = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Pregnancies', 'Age', 'BloodPressure', 'SkinThickness', 'Insulin']

# --- Sidebar ---
st.sidebar.header("1. Data Source")
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
if st.sidebar.button("🔄 Reset Dataset"):
    st.session_state.df = st.session_state.original_df.copy()
    st.session_state.clean_msg = "Dataset Reset to Original."
    st.rerun()

st.title("🚀 ML Pipeline Dashboard")
tabs = st.tabs(["📂 Data & PCA", "📊 EDA", "🧹 Cleaning", "🎯 Features", "🧪 Training", "📈 Performance", "🔮 Prediction"])
target_col = "Outcome"

# --- TAB 1: DATA & PCA ---
with tabs[0]:
    st.subheader("Data Overview")
    st.write("**Current Data Shape:**", st.session_state.df.shape)
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

# --- TAB 2: EDA ---
with tabs[1]:
    st.subheader("Exploratory Data Analysis")
    st.plotly_chart(px.imshow(st.session_state.df.corr(), text_auto=True), use_container_width=True)

# --- TAB 3: CLEANING ---
with tabs[2]:
    st.subheader("Data Cleaning & Engineering")
    
    # Persistent Success/Info Message
    if st.session_state.clean_msg:
        st.info(st.session_state.clean_msg)
    
    col_clean1, col_clean2 = st.columns(2)
    
    with col_clean1:
        st.markdown("### 1. Handle Impossible Zeroes")
        zero_method = st.radio("Select Method:", ["Keep Zeroes", "Delete Rows with Zeroes", "Impute with Median"])
        
        if st.button("Apply Zero-Handling"):
            cols_with_zeroes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            temp_df = st.session_state.df.copy()
            
            if zero_method == "Delete Rows with Zeroes":
                initial_len = len(temp_df)
                for col in cols_with_zeroes:
                    temp_df = temp_df[temp_df[col] != 0]
                st.session_state.df = temp_df.reset_index(drop=True)
                st.session_state.clean_msg = f"✅ Success: {initial_len - len(temp_df)} rows with zeroes removed."
            
            elif zero_method == "Impute with Median":
                for col in cols_with_zeroes:
                    median_val = temp_df[temp_df[col] != 0][col].median()
                    temp_df[col] = temp_df[col].replace(0, median_val)
                st.session_state.df = temp_df
                st.session_state.clean_msg = "✅ Success: Impossible zeroes replaced with medians."
            else:
                st.session_state.clean_msg = "ℹ️ Zeroes retained."
            st.rerun()

    with col_clean2:
        st.markdown("### 2. Outlier Removal")
        outlier_method = st.selectbox("Detection Method:", ["Isolation Forest", "IQR Method"])
        clean_mode = st.checkbox("Stack with current cleaning", value=True)

        if st.button("Clean Outliers"):
            temp_df = st.session_state.df.copy() if clean_mode else st.session_state.original_df.copy()
            initial_shape = temp_df.shape[0]
            
            if outlier_method == "Isolation Forest":
                iso = IsolationForest(contamination=0.05, random_state=42)
                mask = iso.fit_predict(temp_df[st.session_state.active_features])
                temp_df = temp_df[mask == 1]
            else:
                mask = pd.Series([True] * len(temp_df))
                for col in st.session_state.active_features:
                    Q1, Q3 = temp_df[col].quantile(0.25), temp_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask &= (temp_df[col] >= (Q1 - 1.5 * IQR)) & (temp_df[col] <= (Q3 + 1.5 * IQR))
                temp_df = temp_df[mask]
            
            removed = initial_shape - temp_df.shape[0]
            st.session_state.df = temp_df.reset_index(drop=True)
            st.session_state.clean_msg = f"✅ Success: {removed} outliers removed using {outlier_method}."
            st.rerun()

# --- TAB 4: FEATURE ENGINEERING ---
with tabs[3]:
    st.header("Feature Engineering & Selection")
    
    # 1. Selection Method
    selection_method = st.radio(
        "Select Method:", 
        ["All Features", "Variance Threshold", "Information Gain"], 
        horizontal=True
    )
    
    X_temp = st.session_state.df.drop(columns=[target_col])
    y_temp = st.session_state.df[target_col]
    
    # Logic for calculations
    if selection_method == "All Features":
        selected_cols = list(X_temp.columns)
        st.info("All available features are currently active.")

    elif selection_method == "Variance Threshold":
        sel = VarianceThreshold(threshold=0.1).fit(X_temp)
        selected_cols = list(X_temp.columns[sel.get_support()])
        st.caption("Removed features with variance lower than 0.1")

    elif selection_method == "Information Gain":
        # Calculate Mutual Information
        scores = mutual_info_classif(X_temp, y_temp, random_state=42)
        mi_series = pd.Series(scores, index=X_temp.columns).sort_values(ascending=True)
        
        # Display Graph
        st.subheader("Feature Importance (Mutual Information)")
        fig_mi = px.bar(
            mi_series, 
            orientation='h', 
            labels={'value': 'Importance Score', 'index': 'Features'},
            color=mi_series,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_mi, use_container_width=True)
        
        # Select Top 5
        selected_cols = list(mi_series.tail(5).index[::-1])

    # 3. Display Selected Features (JSON Style)
    st.write("**Selected Features:**")
    st.code(f"{selected_cols}", language="json")
    
    # 4. Preview Table
    st.write("**Filtered Data Preview:**")
    st.dataframe(st.session_state.df[selected_cols].head(10), use_container_width=True)
    
    # Update state for training
    st.session_state.active_features = selected_cols

# --- TAB 5: TRAINING ---
with tabs[4]:
    st.subheader("Model Training")
    test_size = st.slider("Test Size (%)", 10, 50, 20)
    k_val = st.number_input("K-Fold (K)", 2, 10, 5)
    
    if st.button("Train Model"):
        # 1. Prepare Data
        X = st.session_state.df[st.session_state.active_features]
        y = st.session_state.df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        
        # 2. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. Model Fitting
        model = RandomForestClassifier(random_state=42) if model_choice == "Random Forest" else LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # 4. Generate Predictions for Metrics
        y_pred = model.predict(X_test_scaled)
        
        # 5. Calculate New Metrics (Precision, Recall, F1)
        # Using 'macro' average to get the unweighted mean of classes
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # 6. Save Everything to Session State
        st.session_state.trained_model = model
        st.session_state.trained_scaler = scaler
        st.session_state.trained_features = st.session_state.active_features
        
        # Standard accuracy
        st.session_state.accuracy = model.score(X_test_scaled, y_test)
        
        # Cross-validation score
        st.session_state.cv_score = cross_val_score(model, scaler.fit_transform(X), y, cv=int(k_val)).mean()
        
        # Store the 3 new metrics
        st.session_state.precision = precision
        st.session_state.recall = recall
        st.session_state.f1 = f1
        
        # Store data for the Confusion Matrix
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        
        st.success("Model trained successfully.")

# --- TAB 6: PERFORMANCE ---
with tabs[5]:
    st.header("📈 Performance Metrics")
    if 'trained_model' in st.session_state:
        # Top Row: Accuracy Metrics
        col1, col2 = st.columns(2)
        col1.metric("Test Accuracy", f"{st.session_state.accuracy:.2%}")
        col2.metric("CV Avg Accuracy", f"{st.session_state.cv_score:.2%}")
        
        st.divider()
        
        # Second Row: Detailed Averages
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Precision", f"{st.session_state.precision:.2%}")
        m2.metric("Avg Recall", f"{st.session_state.recall:.2%}")
        m3.metric("Avg F1 Score", f"{st.session_state.f1:.2%}")
        
        st.divider()
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        fig_cm = px.imshow(
            cm, 
            text_auto=True, 
            labels=dict(x="Predicted", y="Actual"), 
            x=['Non-Diabetic', 'Diabetic'], 
            y=['Non-Diabetic', 'Diabetic'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("Please go to the 'Training' tab and click 'Train Model' first.")

# --- TAB 7: PREDICTION ---
with tabs[6]:
    if 'trained_model' in st.session_state:
        st.subheader("🔮 Predict for New Data")
        input_data = []
        trained_cols = st.session_state.trained_features
        
        cols = st.columns(3)
        for i, feat in enumerate(trained_cols):
            with cols[i % 3]:
                # Determine data type and step based on feature name
                if feat == 'BMI' or feat == 'DiabetesPedigreeFunction':
                    # Float features: Use 0.1 increments
                    val = st.number_input(
                        f"{feat}", 
                        value=float(st.session_state.df[feat].median()),
                        step=0.1,
                        format="%.1f" if feat == 'BMI' else "%.3f"
                    )
                else:
                    # Integer features (BP, Age, Pregnancies, etc.): Use whole numbers
                    val = st.number_input(
                        f"{feat}", 
                        value=int(st.session_state.df[feat].median()),
                        step=1
                    )
                input_data.append(val)
        
        if st.button("Generate Prediction"):
            # Transform input using the scaler from training
            final_input = st.session_state.trained_scaler.transform([input_data])
            prediction = st.session_state.trained_model.predict(final_input)[0]
            probs = st.session_state.trained_model.predict_proba(final_input)[0]
            
            st.divider()
            if prediction == 1:
                st.error(f"### 🚩 Result: Diabetic")
                st.write(f"**Confidence Score:** {probs[1]:.2%}")
            else:
                st.success(f"### ✅ Result: Non-Diabetic")
                st.write(f"**Confidence Score:** {probs[0]:.2%}")
            
            st.progress(float(max(probs)))
    else:
        st.warning("⚠️ Please train the model in the 'Training' tab before making predictions.")