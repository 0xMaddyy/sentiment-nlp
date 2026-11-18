# Sentiment Analysis on Social Media / Product Reviews

Classifies text as positive or negative sentiment using TF-IDF and Logistic Regression.

## Dataset
- Source: [Twitter Sentiment Analysis Dataset](https://github.com/vineetdhanawat/twitter-sentiment-analysis) (1.5M labeled tweets, binary sentiment).
- Used a random sample of 50,000 rows for training speed.
- Not included in this repo due to size. Download it yourself and place it at `data/raw.csv`.

## Pipeline
1. `preprocess.py`: cleans text (removes URLs, mentions, punctuation, lowercases).
2. `train.py`: builds TF-IDF features (unigrams + bigrams, 10k features), trains Logistic Regression.
3. `predict.py`: loads the saved model and predicts sentiment for new text.
4. `app.py`: Streamlit web app for interactive predictions.

## Results
Trained on 39,881 samples, tested on 9,971 held-out samples.

| Metric | Score |
|---|---|
| Accuracy | 77.5% |
| F1-score | 79.4% |
| Precision (Positive) | 78% |
| Recall (Positive) | 81% |

These numbers are from an actual run. Your numbers will vary slightly depending on your random sample and dataset version.

## How to run
```bash
pip install -r requirements.txt
python preprocess.py
python train.py
python predict.py "This product is amazing"
streamlit run app.py
```

## Possible improvements
- Swap Logistic Regression for a fine-tuned DistilBERT model to push F1 past 85%.
- Add a neutral class if your dataset supports 3-way sentiment.
- Handle emojis and slang explicitly for social media text.

## Tech stack
Python, scikit-learn, pandas, Streamlit
