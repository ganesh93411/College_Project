import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score

# Page config
st.set_page_config(page_title="🌟 Bankruptcy Predictor", layout="wide")

# HTML + CSS + JavaScript Styling + Greeting
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            color: #6a0dad;
            text-align: center;
            margin-top: 10px;
        }
        .greeting {
            color: #4B0082;
            font-size: 1.2em;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton > button {
            background-color: #6a0dad;
            color: white;
        }
    </style>
    <script>
        const now = new Date();
        const hour = now.getHours();
        let greeting = "Welcome";
        if (hour < 12) {
            greeting = "Good Morning ☀️";
        } else if (hour < 18) {
            greeting = "Good Afternoon 🌞";
        } else {
            greeting = "Good Evening 🌙";
        }
        window.onload = () => {
            document.getElementById("greeting").innerText = greeting;
        };
    </script>
    <div class='title'>💼 Bankruptcy Prediction App</div>
    <div id='greeting' class='greeting'></div>
""", unsafe_allow_html=True)

# Header image
st.image("https://moretskylaw.com/wp-content/uploads/2018/12/Bankruptcy-1.jpg", use_container_width=True)

# Login Credentials
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}

# Login
def login():
    st.subheader("🔐 Please Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state['authenticated'] = True
            st.success(f"✅ Welcome {username}!")
        else:
            st.error("❌ Invalid login")

# AI Chatbot logic
def chatbot_response(user_input):
    responses = {
        "bankruptcy": "Bankruptcy is a legal process involving a person or business that is unable to repay outstanding debts.",
        "models": "Logistic regression, Random Forest, SVM",
        "logistic regression": "Logistic Regression is a model used for binary classification. It predicts the probability of bankruptcy.",
        "random forest": "Random Forest builds multiple decision trees to improve prediction accuracy.",
        "svm": "SVM is a model that finds a hyperplane to separate classes in the feature space.",
        "avoid bankruptcy": [
            "Ensure regular audits.",
            "Avoid unnecessary debts.",
            "Control expenses.",
            "Improve cash flow.",
            "Plan for uncertainty."
        ],
        "final prediction": "The model uses scaled inputs to compute the probability of bankruptcy. If the score is high, you should take financial actions."
    }

    user_input = user_input.lower()
    if "bankruptcy" in user_input:
        return responses["bankruptcy"]
    elif "models used" in user_input:
        return responses["models"]
    elif "logistic" in user_input:
        return responses["logistic regression"]
    elif "random forest" in user_input:
        return responses["random forest"]
    elif "svm" in user_input:
        return responses["svm"]
    elif "avoid" in user_input:
        return "Ways to avoid bankruptcy:\n- " + "\n- ".join(responses["avoid bankruptcy"])
    elif "assist" in user_input or "final prediction" in user_input:
        return responses["final prediction"]
    else:
        return "Try asking about models like 'logistic regression', or how to avoid bankruptcy."

# AI Chatbot UI
def ai_chatbot_tab():
    st.header("🤖 AI Chatbot Assistant")
    st.write("Ask about bankruptcy, models used, or prediction support.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("💬 Ask a question:")
    if st.button("Ask"):
        if user_input:
            response = chatbot_response(user_input)
            st.session_state.chat_history.append((user_input, response))

    for q, r in reversed(st.session_state.chat_history[-5:]):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI Chatbot:** {r}")

# Dashboard
def show_dashboard():
    tab = st.sidebar.radio("📌 Navigation", [
        "📁 Upload & Visualize", "🧠 Train", "📈 Predict", "📊 Compare", "📋 Evaluate", "🤖 AI Assistant"
    ])

    if tab == "📁 Upload & Visualize":
        st.header("📁 Upload & Explore Dataset")
        file = st.file_uploader("Upload Bankruptcy Data (.xlsx)", type=["xlsx"])
        if file:
            df = pd.read_excel(file, engine='openpyxl')
            st.dataframe(df.head(10))
            st.session_state['data'] = df

            st.subheader("🎨 Feature Distributions")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax, color=np.random.choice(['#8e44ad', '#27ae60', '#3498db']))
                st.pyplot(fig)

            if st.checkbox("Show Correlation Heatmap"):
                fig, ax = plt.subplots()
                sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

    elif tab == "🧠 Train":
        st.header("🧠 Model Training")
        if 'data' not in st.session_state:
            st.warning("📢 Upload dataset first.")
            return

        df = st.session_state['data']
        if 'class' not in df.columns:
            st.error("❌ 'class' column missing.")
            return

        X = df.drop('class', axis=1)
        y = df['class'].map({'bankruptcy': 1, 'non-bankruptcy': 0})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if st.button("🚀 Train Models"):
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                "SVM": SVC(probability=True),
                "KNN": KNeighborsClassifier(n_neighbors=5)
            }
            st.session_state['model_results'] = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                st.session_state['model_results'][name] = {
                    "model": model,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba),
                    "conf_matrix": confusion_matrix(y_test, y_pred),
                    "roc_curve": roc_curve(y_test, y_proba)
                }
                st.success(f"✅ {name} trained!")

            st.session_state['features'] = X.columns.tolist()
            st.session_state['scaler'] = scaler

    elif tab == "📈 Predict":
        st.header("📈 Bankruptcy Prediction")
        if 'model_results' not in st.session_state:
            st.warning("🚨 Train models first.")
            return

        model_choice = st.selectbox("Select Model", list(st.session_state['model_results'].keys()))
        inputs = [st.number_input(f"{feat}", 0.0, 1.0, step=0.1) for feat in st.session_state['features']]
        if st.button("🔮 Predict"):
            model = st.session_state['model_results'][model_choice]['model']
            input_scaled = st.session_state['scaler'].transform(np.array(inputs).reshape(1, -1))
            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]
            st.success("💥 Bankruptcy" if pred == 1 else "✅ No Bankruptcy")
            st.info(f"Confidence Score: {proba:.2f}")

    elif tab == "📊 Compare":
        st.header("📊 Model Comparison")
        if 'model_results' not in st.session_state:
            st.warning("Train the models first.")
            return

        comp_data = [{
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1 Score": res["f1"],
            "ROC AUC": res["roc_auc"]
        } for name, res in st.session_state['model_results'].items()]
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp)

        metric = st.selectbox("Compare by", df_comp.columns[1:])
        fig, ax = plt.subplots()
        sns.barplot(x="Model", y=metric, data=df_comp, palette="magma", ax=ax)
        st.pyplot(fig)

    elif tab == "📋 Evaluate":
        st.header("📋 Model Evaluation")
        model_choice = st.selectbox("Select Model", list(st.session_state['model_results'].keys()))
        res = st.session_state['model_results'][model_choice]

        st.subheader("Classification Report")
        st.text(classification_report(res['y_test'], res['y_pred']))

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(res['conf_matrix'], annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("ROC Curve")
        fpr, tpr, _ = res['roc_curve']
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {res['roc_auc']:.2f}")
        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.legend()
        st.pyplot(fig2)

    elif tab == "🤖 AI Assistant":
        ai_chatbot_tab()

# Main
def main():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if not st.session_state['authenticated']:
        login()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
