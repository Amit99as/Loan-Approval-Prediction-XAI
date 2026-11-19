
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from sklearn.inspection import permutation_importance


# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="wide"
)

st.title("Loan Approval Prediction System")
st.caption("Machine Learning Model Comparison & Prediction")



# 2. LOAD DATASET


@st.cache_data
def load_data():

    df = pd.read_csv("data/loan_approval_dataset.csv")

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Remove extra spaces from categorical values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


try:
    df = load_data()

except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


# 3. DATA PREPROCESSING


features = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

target = "loan_status"

required_columns = features + [target]

if not all(col in df.columns for col in required_columns):

    st.error("Required columns are missing from the dataset.")
    st.stop()


# Select only required columns
df = df[required_columns].copy()

# Encode categorical features
df["education"] = df["education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

df["self_employed"] = df["self_employed"].map({
    "Yes": 1,
    "No": 0
})

# Encode target variable
df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# Handle missing values
df = df.dropna()

X = df[features]
y = df[target].astype(int)

if y.nunique() != 2 or y.value_counts().min() < 2:
    st.error("The dataset must contain sufficient examples of both classes.")
    st.stop()


# 4. TRAIN-TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



# 5. INITIALIZE AND TRAIN MODELS


@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    trained_models = {}
    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[
            :, list(model.classes_).index(1)
        ]

        results.append({

            "Model": name,

            "Accuracy": accuracy_score(
                y_test, predictions
            ),

            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),

            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),

            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),

            "ROC-AUC": roc_auc_score(
                y_test, probabilities
            )
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results).set_index("Model")

    return trained_models, results_df


trained_models, results_df = train_models(
    X_train, y_train, X_test, y_test
)



# 6. DATASET OVERVIEW


st.header(" Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Training Records", len(X_train))
col3.metric("Testing Records", len(X_test))

with st.expander("View Dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# 7. MODEL PERFORMANCE COMPARISON


st.header(" Model Performance Comparison")

st.write(
    "All three models are evaluated using the same "
    "80:20 train-test split."
)

st.dataframe(
    results_df.style.format("{:.3f}"),
    use_container_width=True
)

# Smaller comparison chart
st.subheader("Performance Comparison Chart")

fig, ax = plt.subplots(figsize=(6, 3))

results_df.plot(
    kind="bar",
    ax=ax,
    ylim=(0, 1.05),
    width=0.7
)

ax.set_ylabel("Performance Score")
ax.set_xlabel("ML Models")
ax.set_title("Model Comparison")

ax.legend(
    fontsize=7,
    loc="lower right"
)

ax.tick_params(
    axis="x",
    labelrotation=0,
    labelsize=8
)

plt.tight_layout()

st.pyplot(fig, width="content")

plt.close(fig)


# 8. BEST PERFORMING MODEL


best_model_name = results_df["F1 Score"].idxmax()

st.success(
    f"Best Model based on F1 Score: {best_model_name}"
)


# 9. MODEL SELECTION


st.header(" Select Machine Learning Model")

selected_model_name = st.selectbox(
    "Choose a model:",
    list(trained_models.keys())
)

selected_model = trained_models[selected_model_name]

st.info(f"Selected Model: {selected_model_name}")



# 10. CONFUSION MATRIX


st.subheader("Confusion Matrix")

test_predictions = selected_model.predict(X_test)

fig, ax = plt.subplots(figsize=(4, 3))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    display_labels=["Rejected", "Approved"],
    cmap="Blues",
    ax=ax,
    colorbar=False
)

ax.set_title(selected_model_name, fontsize=10)

plt.tight_layout()

st.pyplot(fig, width="content")

plt.close(fig)



# 11. FEATURE IMPORTANCE


st.subheader("Feature Importance")

importance = permutation_importance(
    selected_model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="f1"
)

importance_df = pd.DataFrame({

    "Feature": features,

    "Importance": importance.importances_mean

}).sort_values(
    "Importance",
    ascending=False
)

# Display top 8 features
top_features = importance_df.head(8).sort_values(
    "Importance"
)

fig, ax = plt.subplots(figsize=(6, 3.5))

ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax.set_xlabel("Mean F1 Score Decrease", fontsize=8)
ax.set_title("Top 8 Important Features", fontsize=10)

ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()

st.pyplot(fig, width="content")

plt.close(fig)



# 12. APPLICANT DETAILS


st.header(" Loan Applicant Details")

with st.form("loan_form"):

    col1, col2 = st.columns(2)

    with col1:

        no_of_dependents = st.number_input(
            "Number of Dependents",
            0, 10, 1
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        income_annum = st.number_input(
            "Annual Income",
            0, 10000000, 500000,
            step=50000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            0, 5000000, 200000,
            step=10000
        )

        loan_term = st.number_input(
            "Loan Term (Months)",
            1, 480, 180
        )

    with col2:

        cibil_score = st.slider(
            "CIBIL Score",
            300, 900, 600
        )

        res_assets = st.number_input(
            "Residential Assets Value",
            0, 50000000, 500000,
            step=100000
        )

        com_assets = st.number_input(
            "Commercial Assets Value",
            0, 50000000, 0,
            step=100000
        )

        lux_assets = st.number_input(
            "Luxury Assets Value",
            0, 50000000, 0,
            step=100000
        )

        bank_assets = st.number_input(
            "Bank Assets Value",
            0, 50000000, 100000,
            step=100000
        )

    submit = st.form_submit_button(
        "Predict Loan Approval"
    )


# 13. LOAN PREDICTION


if submit:

    input_df = pd.DataFrame([{

        "no_of_dependents": no_of_dependents,

        "education": 1 if education == "Graduate" else 0,

        "self_employed": 1 if self_employed == "Yes" else 0,

        "income_annum": income_annum,

        "loan_amount": loan_amount,

        "loan_term": loan_term,

        "cibil_score": cibil_score,

        "residential_assets_value": res_assets,

        "commercial_assets_value": com_assets,

        "luxury_assets_value": lux_assets,

        "bank_asset_value": bank_assets

    }], columns=features)

    prediction = selected_model.predict(input_df)[0]

    probability = selected_model.predict_proba(input_df)[0][
        list(selected_model.classes_).index(1)
    ]

    st.header(" Prediction Result")

    st.write(
        f"Model Used: **{selected_model_name}**"
    )

    if prediction == 1:

        st.success("✅ Loan Approved")

    else:

        st.error("❌ Loan Rejected")

    st.metric(
        "Estimated Approval Probability",
        f"{probability:.2%}"
    )


