import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from collections import Counter
import random

random.seed(42)  # Κλείδωμα για σταθερά αποτελέσματα

# === 1. Φόρτωση δεδομένων και καθαρισμός ===
df = pd.read_csv("video_games_ready_with_dummy.csv")
df = df[df['reviewerID'].notna()]
df = df[df['title'].notna()]
df = df[df['overall'].notna()]
df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')

# Δημιουργία πίνακα TF-IDF για content-based χαρακτηριστικά
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text'])
df['tfidf_index'] = range(len(df))  # αντιστοίχιση index κάθε γραμμής

# === 2. Υβριδική συνάρτηση συστάσεων ===
def hybrid_recommend(user_id, alpha=0.5, top_n=5, min_rating=3):
    user_data = df[df['reviewerID'] == user_id]
    liked_items = user_data[user_data['overall'] >= min_rating]

    if liked_items.empty:
        return pd.DataFrame()

    # Μέσο προφίλ χρήστη από TF-IDF (content-based μέρος)
    liked_indices = liked_items['tfidf_index'].tolist()
    user_profile = np.asarray(tfidf_matrix[liked_indices].mean(axis=0))
    content_sim = cosine_similarity(user_profile, tfidf_matrix).flatten()

    # User-based μέρος (user-item matrix)
    user_item_matrix = df.pivot_table(index='reviewerID', columns='asin', values='overall').fillna(0)
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_vector = user_item_matrix.loc[user_id]
    user_sim = cosine_similarity([user_vector], user_item_matrix)[0]
    user_similarities = pd.Series(user_sim, index=user_item_matrix.index)
    similar_users = user_similarities.drop(index=user_id).sort_values(ascending=False)

    # Ευθυγράμμιση των indexes για αποφυγή σφαλμάτων
    similar_users = similar_users.reindex(user_item_matrix.index).fillna(0)

    # Σταθμισμένη πρόβλεψη αξιολογήσεων από παρόμοιους χρήστες
    weighted_ratings = user_item_matrix.T.dot(similar_users) / (similar_users.sum() + 1e-8)
    weighted_ratings = weighted_ratings.drop(user_item_matrix.columns[user_vector > 0])  # exclude already rated

    # Συνδυασμός User-based και Content-based scores
    hybrid_scores = {}
    for asin in weighted_ratings.index:
        idx = df[df['asin'] == asin].index
        if len(idx) > 0:
            hybrid_scores[asin] = alpha * weighted_ratings[asin] + (1 - alpha) * content_sim[idx[0]]

    # Ταξινόμηση βάσει score
    sorted_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    rec_df = pd.DataFrame(sorted_scores, columns=['asin', 'score']).head(top_n)
    return rec_df

# === 3. Precision / Recall / RMSE ===

# Εύρεση ενεργών χρηστών με ≥6 αξιολογήσεις
user_counts = Counter(df['reviewerID'])
active_users = [user for user, count in user_counts.items() if count >= 6]
sample_users = random.sample(active_users, min(10, len(active_users)))

# Λίστες για συγκέντρωση μετρικών
precision_list, recall_list, actual_ratings, predicted_scores = [], [], [], []

for user in sample_users:
    user_data = df[df['reviewerID'] == user]
    rated_items = user_data[user_data['overall'] > 0]
    if len(rated_items) < 2:
        continue

    # Τελευταίο στοιχείο για test
    test_item = rated_items.iloc[-1]
    test_asin = test_item['asin']
    test_score = test_item['overall']

    # "Αφαίρεση" του test item για να γίνει πρόβλεψη
    df.loc[(df['reviewerID'] == user) & (df['asin'] == test_asin), 'overall'] = 0

    # Εκτέλεση σύστασης
    recs = hybrid_recommend(user, alpha=0.5, top_n=5)
    if recs.empty:
        continue

    # Αξιολόγηση αποτελεσμάτων
    top_asins = recs['asin'].tolist()
    predicted_score = recs[recs['asin'] == test_asin]['score'].values

    # Υπολογισμός precision και recall
    precision = 1 if test_asin in top_asins else 0
    recall = precision / 1  # μόνο 1 relevant item

    precision_list.append(precision)
    recall_list.append(recall)
    if len(predicted_score) > 0:
        actual_ratings.append(test_score)
        predicted_scores.append(predicted_score[0])

# === 4. Τελική εκτύπωση μετρικών ===
avg_precision = np.mean(precision_list) if precision_list else 0
avg_recall = np.mean(recall_list) if recall_list else 0
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_scores)) if predicted_scores else 0

print(f"Hybrid Results")
print(f"Precision@5: {avg_precision:.4f}")
print(f"Recall@5: {avg_recall:.4f}")
print(f"RMSE: {rmse:.4f}")
