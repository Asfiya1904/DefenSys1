import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import plotly.express as px

# ---------- Auth ----------
def login():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "role": None}
    if not st.session_state.auth["logged_in"]:
        st.title("🔐 Welcome to DefenSys Pro")
        with st.form("Login Form"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            role = st.selectbox("Select Role", ["Admin", "Analyst", "Viewer"])
            submit = st.form_submit_button("Login")
            if submit and user and pw:
                st.session_state.auth = {"logged_in": True, "role": role, "user": user}
                st.success(f"Logged in as {user} ({role})")
            else:
                st.error("Please enter credentials.")
        st.stop()

# ---------- Detection ----------
def detect_fraud(df):
    df_numeric = df.select_dtypes(include=[np.number])
    model = IsolationForest(contamination=0.1)
    preds = model.fit_predict(df_numeric)
    df["Risk Score"] = np.round(model.decision_function(df_numeric) * -100, 2)
    df["Severity"] = pd.cut(df["Risk Score"], [-1, 30, 60, 9999], labels=["Low", "Medium", "High"])
    df["Status"] = np.where(preds == -1, "🔴 Suspicious", "🟢 Normal")
    return df

# ---------- UI Pages ----------
def dashboard():
    st.title("📊 Dashboard")
    st.markdown("Upload financial logs or try demo mode. Data will be analyzed using AI anomaly detection.")

    use_demo = st.toggle("🧪 Use Demo Data Instead", value=True)
    df = None

    if use_demo:
        df = pd.read_csv("sample_transaction.csv")
    else:
        uploaded = st.file_uploader("📤 Upload CSV File", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)

    if df is not None:
        result = detect_fraud(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 Threats", (result["Status"] == "🔴 Suspicious").sum())
        col2.metric("🟢 Normal", (result["Status"] == "🟢 Normal").sum())
        col3.metric("📈 Avg Risk", f"{result['Risk Score'].mean():.2f}")

        with st.expander("ℹ️ What is Risk Score?"):
            st.markdown("Risk Score is calculated using an AI model (Isolation Forest) to detect outliers in data. "
                        "Higher score = more unusual behavior = more suspicious.")

        fig = px.pie(result, names="Severity", title="Threat Severity Distribution")
        st.plotly_chart(fig)

        st.dataframe(result, use_container_width=True)

        st.download_button("📥 Download Analysis", result.to_csv(index=False), "defensys_report.csv")

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(result)

def profile():
    st.title("👤 Profile")
    auth = st.session_state.auth
    st.write(f"**Username:** {auth['user']}")
    st.write(f"**Role:** {auth['role']}")
    st.write(f"**Logged in at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def history():
    st.title("📁 History")
    if "history" in st.session_state:
        for i, session in enumerate(reversed(st.session_state.history[-5:]), 1):
            with st.expander(f"Session {i}"):
                st.dataframe(session.head(10))
    else:
        st.info("No past sessions.")

def settings():
    st.title("⚙️ Settings")
    st.checkbox("🔁 Enable auto-refresh (coming soon)")
    st.slider("Sensitivity", 0, 100, 50)

# ---------- App Layout ----------
def main():
    login()
    st.set_page_config(page_title="DefenSys Pro", layout="wide")

    pages = {
        "Dashboard": dashboard,
        "Profile": profile,
        "History": history,
        "Settings": settings
    }

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/567/567203.png", width=50)
        st.title("DefenSys Pro")
        nav = st.radio("📂 Navigation", list(pages.keys()))
        st.markdown("---")
        st.caption("Powered by AI ⚡ | vDemo")

    pages[nav]()

if __name__ == "__main__":
    main()
