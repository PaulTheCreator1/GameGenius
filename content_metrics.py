import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import random
from collections import Counter
random.seed(42)

# === 1. Φόρτωση δεδομένων ===
df = pd.read_csv("video_games_ready_with_dummy.csv")
df = df[df['reviewerID'].notna()]
df = df[df['title'].notna()]
df = df[df['overall'].notna()]
df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text'])
df['tfidf_index'] = range(len(df))

# === 2. Content-based συνάρτηση ===
def recommend_for_user(user_id, top_n=5, min_rating=3):
    user_data = df[df['reviewerID'] == user_id]
    liked_items = user_data[user_data['overall'] >= min_rating]

    if liked_items.empty:
        return pd.DataFrame()

    liked_indices = liked_items['tfidf_index'].tolist()
    user_profile = np.asarray(tfidf_matrix[liked_indices].mean(axis=0))
    similarities = cosine_similarity(user_profile, tfidf_matrix).flatten()
    df['similarity'] = similarities

    seen_asins = user_data['asin'].tolist()
    recommendations = df[~df['asin'].isin(seen_asins)]
    results = recommendations.sort_values(by='similarity', ascending=False)
    results = results[results['similarity'] > 0]

    return results[['asin', 'similarity']].drop_duplicates(subset='asin').head(top_n)

# === 3. Precision/Recall/ RMSE μετρικές ===
user_counts = Counter(df['reviewerID'])
active_users = [user for user, count in user_counts.items() if count >= 6]
sample_users = random.sample(active_users, min(10, len(active_users)))

precision_list, recall_list, actual_ratings, predicted_scores = [], [], [], []

for user in sample_users:
    user_data = df[df['reviewerID'] == user]
    rated_items = user_data[user_data['overall'] > 0]
    if len(rated_items) < 2:
        continue

    test_item = rated_items.iloc[-1]
    test_asin = test_item['asin']
    test_score = test_item['overall']
    df.loc[(df['reviewerID'] == user) & (df['asin'] == test_asin), 'overall'] = 0

    recs = recommend_for_user(user, top_n=5)
    if recs.empty:
        continue

    top_asins = recs['asin'].tolist()
    predicted_score = recs[recs['asin'] == test_asin]['similarity'].values

    precision = 1 if test_asin in top_asins else 0
    recall = precision / 1  # 1 relevant item

    precision_list.append(precision)
    recall_list.append(recall)
    if len(predicted_score) > 0:
        actual_ratings.append(test_score)
        predicted_scores.append(predicted_score[0])

# === 4. Τελικά αποτελέσματα ===
avg_precision = np.mean(precision_list) if precision_list else 0
avg_recall = np.mean(recall_list) if recall_list else 0
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_scores)) if predicted_scores else 0

print(f"Content-based Results")
print(f"Precision@5: {avg_precision:.4f}")
print(f"Recall@5: {avg_recall:.4f}")
print(f"RMSE: {rmse:.4f}")
