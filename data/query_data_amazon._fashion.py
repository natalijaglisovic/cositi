import pandas as pd
import json
import re
from difflib import get_close_matches

# Step 1: Load the review data from local JSONL
print("Loading review data from Amazon_Fashion.jsonl...")
reviews = []
with open("Amazon_Fashion.jsonl", "r") as f:
    for line in f:
        reviews.append(json.loads(line))
reviews = pd.DataFrame(reviews)

# Keep only necessary columns and rename
reviews = reviews[['asin', 'user_id', 'rating', 'verified_purchase', 'helpful_vote' ,'timestamp']]

# Step 2: Filter items with >= 2 interactions
print("Filtering items with at least 2 interactions...")
item_counts = reviews['asin'].value_counts()
popular_items = item_counts[item_counts >= 2].index
reviews = reviews[reviews['asin'].isin(popular_items)]

# Step 3: Filter users with >= 2 interactions
print("Filtering users with at least 2 interactions...")
user_counts = reviews['user_id'].value_counts()
active_users = user_counts[user_counts >= 2].index
filtered_reviews = reviews[reviews['user_id'].isin(active_users)]

# Report number of unique users and items
num_users = filtered_reviews['user_id'].nunique()
num_items = filtered_reviews['asin'].nunique()
print(f"Filtered dataset contains {num_users} unique users and {num_items} unique items.")

# Step 4: Load and filter metadata from local JSONL
print("Loading and filtering metadata from meta_Amazon_Fashion.jsonl...")
target_asins = set(filtered_reviews['asin'].unique())

filtered_metadata = []
with open("meta_Amazon_Fashion.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)
        if record.get("parent_asin") in target_asins:
            filtered_metadata.append(record)

# Convert filtered metadata to DataFrame
metadata_df = pd.DataFrame(filtered_metadata)
print(f"Collected {len(metadata_df)} relevant metadata entries.")

# Step 5: Extract brand/color info from 'details' dict, and directly access title

# List of common colors
common_colors = [
    "black", "white", "red", "blue", "green", "yellow", "pink", "purple",
    "orange", "gray", "grey", "brown", "beige", "navy", "teal", "maroon",
    "gold", "silver", "turquoise", "lavender", "peach", "coral", 
    "lightblue", "darkblue", "lightgreen", "darkgreen", "burgundy", "cream", 
    "mint", "olive", "charcoal", "offwhite", "ivory"
]

def extract_colors_from_title(title):
    if not isinstance(title, str):
        return None
    title_lower = title.lower()
    normalized = re.sub(r'[^a-zA-Z]', ' ', title_lower)
    tokens = normalized.split()
    found_colors = set()
    for token in tokens:
        matches = get_close_matches(token, common_colors, n=1, cutoff=0.85)
        if matches:
            found_colors.add(matches[0])
    if found_colors:
        return "/".join(sorted(c.capitalize() for c in found_colors))
    return None

def extract_brand(record):
    details = record.get("details", {})
    brand = details.get("Brand") if isinstance(details, dict) else None
    if not brand:
        stores = record.get("store", [])
        if isinstance(stores, list) and stores:
            brand = stores[0]
    return brand

# Apply extraction
print("Parsing metadata details...")
metadata_df['brand'] = metadata_df.apply(extract_brand, axis=1)
metadata_df['color'] = metadata_df.apply(
    lambda row: row.get('details', {}).get('Color') if isinstance(row.get('details'), dict) else None,
    axis=1
)
metadata_df['color'] = metadata_df.apply(
    lambda row: row['color'] or extract_colors_from_title(row.get('title', '')),
    axis=1
)

# Step 6: Merge reviews with metadata
print("Merging review data with metadata...")
merged_df = filtered_reviews.merge(
    metadata_df[['parent_asin', 'title', 'brand', 'color']],
    left_on='asin', right_on='parent_asin', how='inner'
)

# Step 6.5: Categorize products based on title keywords
print("Assigning category labels based on title keywords...")

category_keywords = {
    'clothes': ['pants', 'jeans', 'shorts', 'dress', 't-shirt', 'blouse', 'clothes', 'shirt', 'pajamas', 'robe', 'socks', 'top', 'tank', 'cardigan', 'tops', 'capri', 'jacket', 'sweater', 'wool', 'raincoat', 'skirt', 'hoodie'
                'swimsuit', 'bikini', 'sleeve', 'lingerie', 'bra', 'underwear', 'panties', 'tankini'],
    'accessories': ['bags', 'sunglasses', 'belt', 'elastic', 'hairtie', 'gloves', 'scarf', 'tie', 'purse', 'clutch', 'wallet', 'socks', 'hat', 'sock', 'hats', 'handbag', 'tote'],
    'shoes': ['sneakers', 'flats', 'flipflops', 'shoes', 'sandals', 'boot', 'boots'],
    'jewelry': ['necklace', 'bracelet', 'bracelets', 'ring', 'rings', 'jewelry', 'earring', 'earrings', 'watch', 'studs', 'clips', 'pendant', 'chain']
}

def categorize_item(title):
    if not isinstance(title, str):
        return 'unknown'
    title_lower = title.lower()
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', title_lower):
                return category
    return 'unknown'

merged_df['category'] = merged_df['title'].apply(categorize_item)

# Step 7: Select and rename final columns
merged_cleaned_df = merged_df[['title', 'asin', 'user_id', 'brand', 'color',
                               'timestamp', 'rating', 'helpful_vote', 'verified_purchase', 'category']]

merged_cleaned_df.rename(columns={
    'asin': 'itemID',
    'user_id': 'userID',
}, inplace=True)

# Step 8: Save to CSV
merged_cleaned_df.to_csv("amazon_fashion.csv", index=False)
print("Saved cleaned dataset to amazon_fashion.csv")
