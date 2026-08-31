# SMS Spam Classification

A simple Streamlit web app that classifies SMS messages as either spam or not spam using a machine learning model trained on text data.

## Features

- Input any SMS message
- Preprocesses text with lowercasing, tokenization, stopword removal, and stemming
- Uses a TF-IDF vectorizer and trained classifier
- Shows whether the message is spam or not spam

## Project Files

- `app.py` — Streamlit app
- `model.pkl` — trained classification model
- `vectorizer.pkl` — fitted TF-IDF vectorizer

## Run Locally

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install streamlit nltk
```

3. Download NLTK data if needed:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

4. Start the app:

```bash
streamlit run app.py
```

## Example

Enter a message such as:

> Congratulations! You have won a free prize.

The app will classify it as spam.

## Tech Stack

- Python
- Streamlit
- scikit-learn
- NLTK
