
# Sentiment Analysis Web App

This project implements a simple sentiment analysis application using the **Sentiment140** dataset. The app allows users to input a tweet and receive a sentiment prediction (Negative, Neutral, Positive) using a machine learning model. The app is built using **Streamlit** for the web interface and **Logistic Regression** for the model.

## Features
- **Sentiment Analysis** on user-inputted tweets.
- Predicts **Negative**, **Neutral**, or **Positive** sentiment.
- Displays model accuracy based on a preprocessed dataset.
- Clean and simple web interface built with **Streamlit**.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3.  Download the Sentiment140 dataset:
   Since the dataset is not included in the repository due to size constraints, you can download the Sentiment140 dataset from [this link](https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download).
   
   After downloading, place the file in a directory called `dataset/` and rename the file to:
   ```
   dataset/training.1600000.processed.noemoticon.csv
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Once the app is running, you can open it in your web browser.
- Input a tweet in the provided text box and click "Analyze" to see the predicted sentiment.
- The model will classify the tweet as **Negative**, **Neutral**, or **Positive** based on the text input.

## Dataset Information

- The application uses the **Sentiment140** dataset, which contains 1.6 million tweets labeled as **Negative**, **Neutral**, or **Positive**.
- The dataset is not included in the repository because of its large size. Please download the dataset manually (instructions in the Installation section).

## Model

- The model is trained using **Logistic Regression** and text features are extracted using **TF-IDF Vectorization**.
- The app includes basic text preprocessing, including:
  - Removal of URLs, mentions, hashtags.
  - Conversion to lowercase and removal of non-alphabetical characters.
  - Stopword removal using **NLTK**.

## Dependencies

- `streamlit`
- `pandas`
- `nltk`
- `scikit-learn`

You can install all the dependencies by running:
```bash
pip install -r requirements.txt
```

## Project Structure

```bash
.
├── app.py                      # Main app file
├── requirements.txt             # Python dependencies
└── dataset/                     # Directory where the dataset should be placed (not included)
    └── training.1600000.processed.noemoticon.csv  # Sentiment140 dataset (Download manually)
```

## Notes

- If the `dataset/training.1600000.processed.noemoticon.csv` file is not found, the app will not be able to load data or train the model.
- For demonstration purposes, consider using a smaller dataset or sample if the full dataset is not available.

