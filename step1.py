import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from urllib.parse import urljoin

# This function gets news articles from a web page for a specific category
def scrape_news(url, category):
    # Try to get the page content, if it fails print an error and return empty list
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url} — {e}")
        return []

    # Parse the page HTML so we can look for articles
    soup = BeautifulSoup(response.content, 'html.parser')
    data = []

    # Look for article blocks on the page
    article_blocks = soup.select('article, .story-block, .card')

    for item in article_blocks:
        # Find the headline inside the article block
        title_tag = item.find('h3') or item.find('h2') or item.find('h4')
        title = title_tag.get_text(strip=True) if title_tag else "No title"

        # Find the link for the article and make it a full URL
        link_tag = item.find('a', href=True)
        link = urljoin(url, link_tag['href']) if link_tag else ""

        # If we have both title and link, save them
        if title and link:
            data.append({
                "title": title,
                "url": link,
                "category": category,
                "source": url
            })

    # Tell user if no articles were found
    if not data:
        print(f"[!] No articles found for {category} at {url}")

    # Return all the articles found
    return data

# This function runs the whole scraping process for all sites and categories
def main():
    # List of news sites and their category URLs
    news_sites = {
        "abc": {
            "sport": "https://www.abc.net.au/news/sport",
            "lifestyle": "https://www.abc.net.au/news/lifestyle",
            "music": "https://www.abc.net.au/doublej/music-news",
            "finance": "https://www.abc.net.au/news/business"
        },
        "9news": {
            "sport": "https://www.9news.com.au/sport",
            "lifestyle": "https://www.9news.com.au/lifestyle",
            "music": "https://www.9news.com.au/entertainment/music",
            "finance": "https://www.9news.com.au/finance"
        },
        "skynews": {
            "sport": "https://www.skynews.com.au/sport",
            "lifestyle": "https://www.skynews.com.au/lifestyle",
            "music": "https://www.skynews.com.au/lifestyle/music",
            "finance": "https://www.skynews.com.au/business"
        },
        "theaustralian": {
            "sport": "https://www.theaustralian.com.au/sport",
            "lifestyle": "https://www.theaustralian.com.au/lifestyle",
            "music": "https://www.theaustralian.com.au/entertainment/music",
            "finance": "https://www.theaustralian.com.au/business"
        }
    }

    all_articles = []

    # Go through each news site and each category to get articles
    for site_name, categories in news_sites.items():
        for category, url in categories.items():
            print(f"Scraping {category} news from {site_name} - {url}")
            articles = scrape_news(url, category)

            # Add the site name to each article so we know where it came from
            for a in articles:
                a["source_site"] = site_name

            # Collect all articles together
            all_articles.extend(articles)

    # Make a folder called 'data' if it doesn't exist already
    if not os.path.exists('data'):
        os.makedirs('data')

    # Turn the list of articles into a table and save as a CSV file
    df = pd.DataFrame(all_articles)
    csv_path = 'data/news_articles_allsites.csv'
    df.to_csv(csv_path, index=False)

    # Print how many articles were saved and where
    print(f"Scraping done: {len(all_articles)} articles saved to {csv_path}")

# Start the scraping when you run this file
if __name__ == "__main__":
    main()
