import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter

# Load the news articles from the CSV file
df = pd.read_csv("data/news_articles_allsites.csv")

# Make sure there is a 'content' column; if not, create an empty one
if "content" not in df.columns:
    df["content"] = ""

# Join title and content into one text for processing
df["text"] = df["title"].fillna("") + " " + df["content"].fillna("")

# This dictionary has keywords for each category
category_keywords = {
    "sports": [
        "sport", "game", "player", "team", "score", "match", "league", "football",
        "cricket", "tennis", "basketball", "soccer", "rugby", "olympics",
        "athlete", "coach", "tournament", "season", "goal", "win", "loss", "medal",
        "stadium", "championship", "referee", "injury", "fifa"
    ],
    "lifestyle": [
        "lifestyle", "health", "travel", "food", "fashion", "wellness", "culture",
        "home", "family", "fitness", "beauty", "recipe", "diet", "mental health",
        "relationship", "garden", "hobby", "leisure", "holiday", "vacation",
        "yoga", "spa", "makeup", "shopping"
    ],
    "music": [
        "music", "album", "concert", "song", "band", "artist", "festival", "dj",
        "track", "melody", "genre", "single", "playlist", "tour", "orchestra",
        "composer", "vocal", "pop", "rock", "jazz", "hip-hop", "classical",
        "recording", "producer", "chart"
    ],
    "finance": [
        "finance", "market", "stock", "investment", "bank", "economy", "business",
        "trade", "money", "share", "inflation", "revenue", "profit", "loss",
        "fund", "currency", "cryptocurrency", "bonds", "tax", "loan", "interest",
        "economics", "budget", "startup", "merger", "acquisition", "IPO",
        "dividend", "portfolio", "finance minister"
    ],
}

# This function checks which category fits best for the article text
def predict_category(text):
    text = text.lower()
    scores = {cat: sum(text.count(word) for word in keywords) for cat, keywords in category_keywords.items()}
    max_score = max(scores.values())
    if max_score == 0:
        return "other"  # No matching keywords means no category
    return max(scores, key=scores.get)

# Apply the function to each article to assign categories
df["predicted_category"] = df["text"].apply(predict_category)

# Remove articles that have no category assigned (marked as 'other')
df = df[df["predicted_category"] != "other"].reset_index(drop=True)

# Print how many articles we have in each category
print("Category counts after removing 'other':")
print(df["predicted_category"].value_counts())

# Now we find duplicate or very similar articles

# Convert all texts to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df["text"])

# Use DBSCAN clustering to group similar articles together
dbscan = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
df["cluster_id"] = dbscan.fit_predict(X)

# Count how many articles belong to duplicate groups (cluster_id not -1 and count >1)
cluster_counts = Counter(df["cluster_id"])
duplicate_articles_count = sum(count for cid, count in cluster_counts.items() if cid != -1 and count > 1)

# Print total duplicates found
print(f"Total duplicate articles: {duplicate_articles_count}")

# Save the updated data with categories and clusters to a new CSV file
df.to_csv("data/news_articles_categorized_deduped_filtered.csv", index=False)
print("Saved categorized and deduplicated articles.")
