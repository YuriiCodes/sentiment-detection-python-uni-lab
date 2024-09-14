import streamlit as st
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re


# Download stopwords from nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Cache the loading of data so it runs once (use st.cache_data for data-related operations)
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/training.1600000.processed.noemoticon.csv", encoding='ISO-8859-1', header=None)
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    # Mapping: 0 = negative, 2 = neutral, 4 = positive -> 0 = negative, 1 = neutral, 2 = positive
    df['target'] = df['target'].replace({0: 0, 2: 1, 4: 2})
    return df


# Cache the model and vectorizer after they are trained (use st.cache_resource for model-related operations)
@st.cache_resource
def train_model():
    df = load_data()

    # Clean the tweet text
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
        text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Remove hashtags
        text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetical characters
        text = text.lower()  # Lowercase
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
        return ' '.join(tokens)

    df['clean_text'] = df['text'].apply(clean_text)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['target'], test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train the model
    model = LogisticRegression(multi_class='ovr')
    model.fit(X_train_vec, y_train)

    # Calculate accuracy
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy


# Load the model and vectorizer
model, vectorizer, accuracy = train_model()

# Streamlit UI
st.title("Sentiment Analysis")

st.write("Trained on the Sentiment140 Dataset (Negative, Neutral, Positive)")

# Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Text input from user
user_input = st.text_area("Enter a tweet to analyze sentiment:")

if st.button("Analyze"):
    if user_input:
        # Clean and vectorize user input
        def clean_input_text(text):
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
            text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Remove hashtags
            text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetical characters
            text = text.lower()  # Lowercase
            tokens = text.split()
            tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
            return ' '.join(tokens)


        clean_input = clean_input_text(user_input)
        input_vectorized = vectorizer.transform([clean_input])

        # Predict sentiment (0 = Negative, 1 = Neutral, 2 = Positive)
        prediction = model.predict(input_vectorized)[0]

        # Map prediction to sentiment
        sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}

        # Display result
        st.write(f"Sentiment: {sentiment[prediction]}")
    else:
        st.write("Please enter a tweet.")

# Show some examples from the dataset
st.write("Example tweets from the dataset:")
st.write(load_data()[['text', 'target']].head())

