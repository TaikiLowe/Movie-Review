import nltk 
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet', 'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])

# Download the dictionary data (only needs to be run once)
nltk.download('vader_lexicon') 

sia = SentimentIntensityAnalyzer()

text_data = df.astype(str).values.flatten()
cutoff = len(text_data) // 1000 
new_text = text_data[:cutoff]
reviews = list(new_text)

print("MOVIE REVIEW SENTIMENT REPORT:")

for i, review in enumerate(reviews, 1):
    scores = sia.polarity_scores(review)
    compound = scores['compound']
    
    # Classify based on thresholds
    if compound >= 0.05:
        sentiment = "POSITIVE"
    elif compound <= -0.05:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
        
    print(f"\nReview #{i}: \"{review}\"")
    print(f"Score: {compound} | Verdict: {sentiment}")
