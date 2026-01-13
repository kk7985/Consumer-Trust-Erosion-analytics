!pip install ntlk
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import plotly.express as px

st.set_page_config(page_title="Trust & Churn Analytics", layout="wide")
st.title("Customer Trust Erosion & Churn Prediction")

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
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df["Trust_Erosion"] = (df["rating"] <= 2).astype(int)
    df["clean_text"] = df["text"].apply(clean_text_fn)
    df["sentiment_score"] = df["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])

    st.subheader("Trust & Sentiment Insights")
    col1, col2 = st.columns(2)

    with col1:
        fig_rating = px.histogram(df, x="rating", 
                                   title="Distribution of Ratings",
                                   color="Trust_Erosion",
                                   color_discrete_map={1: '#EF553B', 0: '#636EFA'},
                                   labels={"Trust_Erosion": "Trust Erosion (Rating <= 2)"})
        st.plotly_chart(fig_rating, use_container_width=True)

    with col2:
        fig_sent = px.box(df, x="Trust_Erosion", y="sentiment_score",
                          title="Sentiment Score: Healthy vs Trust Erosion",
                          color="Trust_Erosion",
                          labels={"Trust_Erosion": "Trust Erosion (1=Yes, 0=No)"})
        st.plotly_chart(fig_sent, use_container_width=True)

    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    trend_data = df.groupby('month')['Trust_Erosion'].sum().reset_index()
    fig_trend = px.line(trend_data, x='month', y='Trust_Erosion', 
                        title="Trend of Negative Reviews (Rating <= 2) Over Time",
                        markers=True, line_shape="spline")
    st.plotly_chart(fig_trend, use_container_width=True)


    latest_date = df["timestamp"].max()
    df["recency"] = (latest_date - df.groupby("user_id")["timestamp"].transform("max")).dt.days
    
    user_df = df.groupby("user_id").agg({
        "Trust_Erosion": "max",
        "sentiment_score": "mean",
        "recency": "max"
    }).reset_index()
    user_df["churn"] = (user_df["recency"] > 90).astype(int)

    X = user_df[["Trust_Erosion", "sentiment_score", "recency"]]
    y = user_df["churn"]

    if len(y.unique()) > 1:
        model = LogisticRegression().fit(X, y)
        
        st.divider()
        st.subheader("Predict Churn Risk")
        c1, c2, c3 = st.columns(3)
        with c1: in_trust = st.selectbox("Low Rating History?", [0, 1])
        with c2: in_sent = st.slider("Sentiment", -1.0, 1.0, 0.0)
        with c3: in_recency = st.number_input("Days Inactive", 0, 365, 30)
        
        if st.button("Calculate Churn Probability"):
            prob = model.predict_proba([[in_trust, in_sent, in_recency]])[0][1]
            st.write(f"### Result: {prob:.1%} Risk of Churn")
            st.progress(prob)

else:

    st.info("Upload a CSV file to see the interactive graphs and predictions.")
