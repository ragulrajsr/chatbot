import pandas as pd

# Load clustered and categorized data from Step 2
df = pd.read_csv("data/news_articles_categorized_deduped_filtered.csv")

# Define priority keywords that boost a highlight's importance
priority_keywords = ["breaking", "urgent", "exclusive", "alert", "update", "important"]

# Filter noise articles (cluster_id = -1) because they have no duplicates (single story)
# Optionally you can also include noise articles as single-article stories
clusters = df[df["cluster_id"] != -1].copy()

# Group articles by cluster to form stories
grouped = clusters.groupby("cluster_id")

highlights = []

for cluster_id, group in grouped:
    category = group["predicted_category"].iloc[0]
    
    # Concatenate all titles in the cluster
    combined_titles = " ".join(group["title"].astype(str).tolist()).lower()
    
    # Count how many priority keywords appear in combined titles
    keyword_score = sum(combined_titles.count(word) for word in priority_keywords)
    
    # Frequency score = number of articles in cluster
    frequency_score = len(group)
    
    # Total score: simple weighted sum (you can tweak weights)
    total_score = keyword_score * 2 + frequency_score
    
    # Pick a representative title (e.g., longest title in cluster)
    rep_title = max(group["title"].astype(str), key=len)
    
    highlights.append({
        "cluster_id": cluster_id,
        "category": category,
        "representative_title": rep_title,
        "keyword_score": keyword_score,
        "frequency_score": frequency_score,
        "total_score": total_score,
        "source_count": frequency_score
    })

highlights_df = pd.DataFrame(highlights)

# For each category, pick top 5 highlights by total_score
top_highlights = highlights_df.groupby("category").apply(
    lambda x: x.sort_values("total_score", ascending=False).head(5)
).reset_index(drop=True)

print(top_highlights[["category", "representative_title", "total_score", "source_count"]])

# Save top highlights for later UI or chatbot use
top_highlights.to_csv("data/news_highlights.csv", index=False)
print("Saved top news highlights to data/news_highlights.csv")
