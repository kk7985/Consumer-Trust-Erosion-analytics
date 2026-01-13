import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px


@st.cache_resource
def load_resources():
    nltk.download("stopwords")
    nltk.download("vader_lexicon")
    return set(stopwords.words("english")), SentimentIntensityAnalyzer()

stop_words, sia = load_resources()

def clean_text_fn(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)


    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["Trust_Erosion"] = (df["rating"] <= 2).astype(int)

    df["clean_text"] = df["text"].apply(clean_text_fn)
    df["sentiment_score"] = df["clean_text"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

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


    latest_date = df["timestamp"].max()
    df["recency"] = (
        latest_date - df.groupby("user_id")["timestamp"].transform("max")
    ).dt.days

    user_df = df.groupby("user_id").agg({
        "Trust_Erosion": "max",
        "sentiment_score": "mean",
        "recency": "max"
    }).reset_index()


    user_df["churn"] = (
        (user_df["recency"] > 90) &
        (user_df["sentiment_score"] < -0.2)
    ).astype(int)

    X = user_df[["Trust_Erosion", "sentiment_score", "recency"]]
    y = user_df["churn"]

    if y.nunique() > 1:
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_scaled, y)

        st.divider()
        st.subheader("Churn Risk Prediction")

        c1, c2, c3 = st.columns(3)
        with c1:
            in_trust = st.selectbox("Low Rating History?", [0, 1])
        with c2:
            in_sent = st.slider("Sentiment", -1.0, 1.0, -0.5, 0.01)
        with c3:
            in_recency = st.number_input("Days Inactive", 0, 365, 30)

        if st.button("Calculate Churn Probability"):
            user_scaled = scaler.transform(
                [[in_trust, in_sent, in_recency]]
            )

            prob = model.predict_proba(user_scaled)[0][1]
            risk = prob * 100

            # Business-safe minimum (no fake zero)
            if in_trust == 1 and in_sent < 0:
                risk = max(risk, 1)

            st.warning(f"### Result: {risk:.2f}% Risk of Churn")
            st.progress(min(risk / 100, 1.0))

    else:
        st.warning("Not enough churn variation in data to train model.")

else:
    st.info("Upload a CSV file to see analytics and churn prediction.")




