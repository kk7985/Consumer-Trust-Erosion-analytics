import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

# -------------------------
# Resource loading
# -------------------------
@st.cache_resource
def load_resources():
    nltk.download("stopwords")
    nltk.download("vader_lexicon")
    return set(stopwords.words("english")), SentimentIntensityAnalyzer()

stop_words, sia = load_resources()

# -------------------------
# Text cleaning
# -------------------------
def clean_text_fn(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# -------------------------
# Outlier removal
# -------------------------
def hard_filters(df):
    df = df[
        (df["rating"].between(1, 5)) &
        (df["sentiment_score"].between(-1, 1)) &
        (df["recency"].between(0, 365))
    ]
    return df

def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# -------------------------
# UI
# -------------------------
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------------------------
    # Basic validation
    # -------------------------
    required_cols = {"user_id", "timestamp", "rating", "text"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain: user_id, timestamp, rating, text")
        st.stop()

    # -------------------------
    # Preprocessing
    # -------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "rating", "text"])

    # Trust erosion
    df["Trust_Erosion"] = (df["rating"] <= 2).astype(int)

    # NLP
    df["clean_text"] = df["text"].apply(clean_text_fn)
    df["sentiment_score"] = df["clean_text"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    # -------------------------
    # Visual Analytics
    # -------------------------
    st.subheader("Trust Erosion & Sentiment Insights")

    col1, col2 = st.columns(2)
    with col1:
        fig_rating = px.histogram(
            df,
            x="rating",
            color="Trust_Erosion",
            title="Distribution of Ratings",
            labels={"Trust_Erosion": "Trust Erosion (Rating ≤ 2)"}
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    with col2:
        fig_sent = px.box(
            df,
            x="Trust_Erosion",
            y="sentiment_score",
            title="Sentiment vs Trust Erosion",
            labels={"Trust_Erosion": "Trust Erosion (1 = Yes)"}
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # Trend
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    trend = df.groupby("month")["Trust_Erosion"].sum().reset_index()

    fig_trend = px.line(
        trend,
        x="month",
        y="Trust_Erosion",
        title="Negative Review Trend Over Time",
        markers=True
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # -------------------------
    # Recency feature
    # -------------------------
    latest_date = df["timestamp"].max()
    df["recency"] = (
        latest_date - df.groupby("user_id")["timestamp"].transform("max")
    ).dt.days

    # -------------------------
    # Aggregation (user-level)
    # -------------------------
    user_df = df.groupby("user_id").agg({
        "Trust_Erosion": "max",
        "sentiment_score": "mean",
        "recency": "max"
    }).reset_index()

    # -------------------------
    # Outlier cleaning
    # -------------------------
    user_df = hard_filters(user_df)
    user_df = remove_outliers_iqr(user_df, ["sentiment_score", "recency"])

    # -------------------------
    # Churn labeling (logic-based, not fake)
    # -------------------------
    user_df["churn"] = (
        (user_df["recency"] > 60) &
        (user_df["sentiment_score"] < -0.3) |
        (user_df["Trust_Erosion"] == 1)
    ).astype(int)

    # -------------------------
    # Model features
    # -------------------------
    X = user_df[["Trust_Erosion", "sentiment_score", "recency"]]
    y = user_df["churn"]

    if y.nunique() > 1:

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(class_weight="balanced")
        model.fit(X_train, y_train)

        st.divider()
        st.subheader("Churn Risk Prediction")

        c1, c2, c3 = st.columns(3)
        with c1:
            in_trust = st.selectbox("Low Rating History?", [0, 1])
        with c2:
            in_sent = st.slider("Sentiment", -1.0, 1.0, -0.4, 0.01)
        with c3:
            in_recency = st.number_input("Days Inactive", 0, 365, 45)

        if st.button("Calculate Churn Probability"):

            user_scaled = scaler.transform(
                [[in_trust, in_sent, in_recency]]
            )

            prob = model.predict_proba(user_scaled)[0][1]
            risk = float(prob * 100)

            # floor logic (no fake zeros)
            if in_trust == 1 and in_sent < 0:
                risk = max(risk, 2.5)

            st.error(f"### {risk:.2f}% Churn Risk")
            st.progress(min(risk / 100, 1.0))

    else:
        st.warning("Not enough churn variation in data to train model.")

else:
    st.info("Upload a CSV file to see analytics and churn prediction.")




