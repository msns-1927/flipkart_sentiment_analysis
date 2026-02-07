# Flipkart Sentiment Analysis | End to End ML Project 🛒
**🚀 Project Overview :**

Developed and deployed an end-to-end Sentiment Analysis system that classifies Flipkart product reviews as Positive or Negative using Machine Learning and Natural Language Processing techniques. The trained model was deployed as a Streamlit web application on AWS EC2, enabling real-time sentiment prediction.

--------------------

**🧩 Problem Statement :**

Online e-commerce platforms like Flipkart receive thousands of customer reviews every day. Manually analyzing these reviews to understand customer satisfaction and product issues is time-consuming and inefficient. There is a need for an automated system that can accurately analyze review text and determine the overall sentiment expressed by customers.

The objective of this project is to build and deploy a machine learning–based sentiment analysis system that classifies Flipkart product reviews as Positive or Negative. The system processes raw review text using Natural Language Processing techniques, converts it into numerical features, and predicts sentiment in real time through a deployed web application. This helps businesses quickly gain insights into customer feedback and make data-driven decisions.

--------------------

**🔑 Key Challenges Addressed :**

- **Noisy and unstructured text data:**
   Handled real-world review text containing irrelevant phrases, punctuation, and inconsistencies using effective NLP preprocessing techniques.

- **Feature representation of text data:**
   Converted raw text into meaningful numerical features using TF-IDF vectorization to enable machine learning models to process textual input.

- **Model selection and evaluation:**
   Trained and compared multiple machine learning models and selected the best-performing one based on F1-score to ensure balanced performance.

- **Consistency between training and deployment:**
   Ensured the same preprocessing steps and TF-IDF vectorizer were used during both model training and real-time prediction.

- **Real-time sentiment prediction:**
   Integrated the trained model into a Streamlit web application to provide instant sentiment predictions for user-provided reviews.

- **Deployment on cloud infrastructure:**
   Successfully deployed the application on AWS EC2, addressing challenges related to environment setup, dependency management, and network access.

--------------------

**📂 Dataset Description :**

The dataset consists of 8,518 customer reviews collected from Flipkart product listings. Each record represents a user’s review along with associated metadata such as rating, review text, and engagement metrics. The dataset is used to perform sentiment analysis by classifying customer opinions as positive or negative.

**🧾 Dataset Structure :**

The dataset contains 8 columns, described below:

| Column Name | Description |
| :---: | :---: |
| Reviewer Name |	Name or identifier of the customer who posted the review |
| Review Title | Short summary or heading of the review |
| Place of Review | Location of the reviewer (if available) |
| Up Votes | Number of users who found the review helpful |
| Down Votes | Number of users who disliked the review |
| Month | Month and year when the review was posted |
| Review text | Full textual content of the customer review |
| Ratings | Product rating given by the user (1 to 5 scale) |


**🧠 Usage in This Project :**

- The Review text column is used as the primary input for sentiment analysis.

- The Ratings column is used to generate sentiment labels:

  - Ratings ≥ 4 → Positive sentiment

  - Ratings ≤ 2 → Negative sentiment

  - Neutral ratings (3) are excluded to improve classification clarity.

- Other metadata columns provide contextual information but are not directly used for model training.

**⚠️ Data Characteristics :**

- The dataset contains real-world noisy text, including spelling variations, punctuation, emojis, and extra phrases such as “READ MORE”.

- Some fields contain missing values, which were handled during preprocessing.

- Reviews vary in length, making text preprocessing essential before modeling.

**🎯 Purpose of the Dataset :**

This dataset enables the development of a machine learning–based sentiment analysis system that can automatically interpret customer feedback and classify product reviews as positive or negative, supporting data-driven decision-making for e-commerce platforms.

--------------------

**🛠️ Methodology :**

The project follows a systematic, end-to-end machine learning workflow to perform sentiment analysis on Flipkart product reviews.

**1️⃣ Data Collection :**

The dataset consists of customer reviews collected from Flipkart, including review text and corresponding ratings. These ratings serve as the basis for deriving sentiment labels.

**2️⃣ Data Preprocessing :**

Raw review text was cleaned to remove noise and inconsistencies. This included:

- Converting text to lowercase

- Removing URLs, punctuation, and special characters

- Eliminating irrelevant phrases such as “READ MORE”

- Removing stopwords and applying lemmatization

This step ensures that only meaningful textual information is retained for analysis.

**3️⃣ Sentiment Labeling :**

Sentiment labels were generated from product ratings:

- Ratings 4 and 5 were labeled as Positive

- Ratings 1 and 2 were labeled as Negative

- Neutral reviews (rating 3) were excluded to improve classification clarity

**4️⃣ Feature Extraction :**

The cleaned review text was transformed into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization, which captures the importance of words while reducing the influence of commonly occurring terms.

**5️⃣ Model Training and Evaluation :**

Multiple machine learning models were trained and evaluated, including:

- Logistic Regression

- Naive Bayes

- Support Vector Machine

- Random Forest

Models were compared using evaluation metrics such as Precision, Recall, and F1-score, and the best-performing model was selected based on F1-score.

**6️⃣ Model Selection and Saving :**

The final selected model and the trained TF-IDF vectorizer were saved using joblib to ensure consistency between training and deployment environments.

**7️⃣ Application Development :**

A Streamlit web application was developed to provide a user-friendly interface where users can input product reviews and receive real-time sentiment predictions.

**8️⃣ Deployment :**

The application was deployed on AWS EC2 using a Python virtual environment. Necessary dependencies were installed, security group rules were configured, and the application was made accessible via the instance’s public IP address.

**✅ Outcome :**

This methodology ensures a reliable, scalable, and deployment-ready sentiment analysis system that processes real-world review data and delivers accurate sentiment predictions in real time.

--------------------

**🔍 Observations :**

- Customer reviews contain highly unstructured and noisy text, making preprocessing an essential step for effective sentiment analysis.

- Reviews with higher ratings (4–5) predominantly express positive sentiment, while lower ratings (1–2) clearly indicate dissatisfaction, validating the use of ratings for sentiment labeling.

- TF-IDF vectorization proved effective in capturing important words and reducing the impact of commonly occurring terms.

- Among the evaluated machine learning models, Random Forest achieved the best performance based on F1-score, indicating a good balance between precision and recall.

- Excluding neutral reviews (rating 3) improved model clarity and reduced ambiguity during classification.

- The deployed Streamlit application provided real-time sentiment predictions, demonstrating the practical usability of the trained model.

- Cloud deployment on AWS EC2 confirmed that the model performs consistently outside the local development environment.

--------------------

**⚠️ Limitations :**

- The model performs binary sentiment classification (Positive/Negative) and does not handle neutral or mixed sentiments.

- Sentiment labels are derived from ratings, which may not always perfectly reflect the actual sentiment expressed in the review text.

- The model relies on traditional ML with TF-IDF and may not capture deeper contextual meaning, sarcasm, or implicit sentiment.

- Performance may vary for very short reviews or reviews containing slang, emojis, or spelling errors.

- The system is trained on Flipkart-specific data, so performance on reviews from other platforms may require retraining.

- The deployed application does not currently support batch predictions or multilingual reviews.

--------------------

**🚀 Future Improvements :**

- Extend the system to support multi-class sentiment classification, including neutral and mixed sentiments.

- Incorporate deep learning models such as LSTM, BERT, or other transformer-based architectures to capture contextual meaning and sarcasm.

- Add multilingual support to analyze reviews written in different languages.

- Enable batch sentiment analysis for processing large volumes of reviews simultaneously.

- Improve preprocessing to better handle emojis, slang, and informal language commonly used in customer reviews.

- Integrate a database or dashboard to store predictions and visualize sentiment trends over time.

- Deploy the application using Docker and CI/CD pipelines for improved scalability and maintainability.

--------------------

**🧰 Tools and Technologies Used :**

- **Programming Language:** Python

- **Libraries & Frameworks:** Scikit-learn, NLTK, Pandas, NumPy

- **Feature Extraction:** TF-IDF Vectorization

- **Model Serialization:** Joblib

- **Web Framework:** Streamlit

- **Cloud Platform:** AWS EC2

- **Development Environment:** Jupyter Notebook

- **Version Control:** GitHub

--------------------

**🔑 Key Takeaway :**

This project demonstrates the complete lifecycle of a machine learning solution—from cleaning real-world text data and selecting the best model using evaluation metrics to deploying the final model as a scalable web application on the cloud. It highlights the importance of consistent preprocessing, data-driven model selection, and practical deployment for building production-ready ML systems.

--------------------

**⚙️ Installation :**

Follow the steps below to set up and run the project locally or on a cloud server.

**1️⃣ Clone the Repository :**

```bash
git clone <your-github-repo-url>
cd flipkart_sentiment
```
**2️⃣ Create and Activate Virtual Environment :**

For Linux / macOS / AWS EC2:
```bash
python3 -m venv venv
source venv/bin/activate
```
For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
**3️⃣ Install Required Dependencies :**
```bash
pip install -r requirements.txt
```
**4️⃣ Run the Application :**
```bash
streamlit run app.py
```

**5️⃣ Access the Application :**

Open your browser and navigate to:
```arduino
http://localhost:8501
```
(For AWS EC2 deployment, use http://<EC2_PUBLIC_IP>:8501)

--------------------

**✅ Notes :**

- Ensure Python 3.8 or above is installed.

- Make sure the ``` models/ ``` directory contains the saved model and TF-IDF vectorizer files.

- For cloud deployment, ensure the required port (8501) is open in the security group.
