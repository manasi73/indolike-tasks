import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the CSV (check exact filename matches your file)
df = pd.read_csv("amazon_product_reviews.csv")

# Print first 5 rows to check
print(df.head())
# Check the column names
print("Columns:", df.columns)

# Print first few review texts
print("\nSample reviews:")
print(df['Review Text'].head())


# Function to get sentiment label
def get_sentiment(text):
    blob = TextBlob(str(text))  # Convert to string in case of NaN
    polarity = blob.sentiment.polarity

    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'


# Apply the function to each review
df['Sentiment'] = df['Review Text'].apply(get_sentiment)

# Show some results
print(df[['Review Text', 'Sentiment']].head())
# Count how many of each sentiment
sentiment_counts = df['Sentiment'].value_counts()
print("\nSentiment Counts:\n", sentiment_counts)

# Count the sentiments
sentiment_counts = df['Sentiment'].value_counts()

# Bar Chart
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['green', 'orange', 'red'])

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
# Pie Chart
plt.figure(figsize=(6, 6))
colors = ['green', 'orange', 'red']
labels = sentiment_counts.index
sizes = sentiment_counts.values

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("Sentiment Distribution (%)")
plt.axis('equal')  # Makes it a perfect circle

plt.tight_layout()
plt.show()
df.to_csv("sentiment_results.csv", index=False)

