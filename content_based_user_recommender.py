import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# === 1. Φόρτωση dataset ===
df = pd.read_csv("video_games_ready.csv")
df = df[df['reviewerID'].notna()]
df = df[df['title'].notna()]
df = df[df['overall'].notna()]

# === 2. Συνδυασμός τίτλου + περιγραφής ===
df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')

# === 3. TF-IDF στο κείμενο ===
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text'])
df['tfidf_index'] = range(len(df))

# === 4. Συνάρτηση προτάσεων για χρήστη ===
def recommend_for_user(user_id, top_n=5, min_rating=3):
    user_data = df[df['reviewerID'] == user_id]
    liked_items = user_data[user_data['overall'] >= min_rating]

    if liked_items.empty:
        return f"Ο χρήστης '{user_id}' δεν έχει αξιολογήσει τίποτα θετικά."

    liked_indices = liked_items['tfidf_index'].tolist()
    user_profile = np.asarray(tfidf_matrix[liked_indices].mean(axis=0))

    similarities = cosine_similarity(user_profile, tfidf_matrix).flatten()
    df['similarity'] = similarities

    seen_asins = user_data['asin'].tolist()
    recommendations = df[~df['asin'].isin(seen_asins)]
    results = recommendations.sort_values(by='similarity', ascending=False)

    # Κρατάμε μόνο προτάσεις με similarity > 0
    results = results[results['similarity'] > 0]

    return results[['asin', 'title', 'similarity']].drop_duplicates(subset='asin').head(top_n).reset_index(drop=True)

# === 5. Επιλογή χρήστη με θετική αξιολόγηση ===
positive_users = df[df['overall'] >= 3]['reviewerID'].unique().tolist()
if not positive_users:
    print("Κανένας χρήστης δεν έχει αξιολογήσει κάτι θετικά.")
else:
    sample_user = random.choice(positive_users)
    print(f"Προτάσεις για χρήστη: {sample_user}")
    print(recommend_for_user(sample_user, top_n=5, min_rating=3))
