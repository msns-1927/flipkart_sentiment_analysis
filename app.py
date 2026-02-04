import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("models/best_sentiment_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# App title
st.set_page_config(page_title="Flipkart Review Sentiment Analysis", layout="centered")
st.title("🛒 Flipkart Product Review Sentiment Analysis")
st.write("Enter a product review to predict whether the sentiment is **Positive** or **Negative**.")

# Text input
user_review = st.text_area("✍️ Enter your review here:")

# Predict button
if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review text.")
    else:
        # Transform input text
        review_vector = vectorizer.transform([user_review])

        # Predict sentiment
        prediction = model.predict(review_vector)[0]

        # Display result
        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")