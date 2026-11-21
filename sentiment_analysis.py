import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import nltk; nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

reviews = ["Amazing product, totally worth it!", "Worst purchase ever, broke in a day",
           "It's okay, nothing special", "Battery life is fantastic", "Delivery was late and packaging was bad",
           "Loved the camera quality", "Average performance for the price", "Terrible customer support"]
df = pd.DataFrame({'text': reviews})

sia = SentimentIntensityAnalyzer()
df['score'] = df.text.apply(lambda t: sia.polarity_scores(t)['compound'])
df['sentiment'] = df.score.apply(lambda s: 'Positive' if s >= .05 else 'Negative' if s <= -.05 else 'Neutral')

print(df)
sns.countplot(x='sentiment', data=df, palette='Set2'); plt.title('Sentiment Distribution'); plt.show()