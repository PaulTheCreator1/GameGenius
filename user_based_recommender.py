import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from collections import Counter
import random

random.seed(42)  # Κλείδωμα για αναπαραγωγιμότητα των αποτελεσμάτων

# === 1. Φόρτωση και φιλτράρισμα δεδομένων ===
df = pd.read_csv("video_games_ready_with_dummy.csv")
df = df[df['reviewerID'].notna()]    # Διαγραφή εγγραφών χωρίς userID
df = df[df['asin'].notna()]          # Διαγραφή εγγραφών χωρίς προϊόν
df = df[df['overall'].notna()]       # Διαγραφή εγγραφών χωρίς βαθμολογία

# Επιλογή χρηστών με τουλάχιστον 6 αξιολογήσεις
user_counts = Counter(df['reviewerID'])
active_users = [user for user, count in user_counts.items() if count >= 6]

print(f"Εντοπίστηκαν {len(active_users)} ενεργοί χρήστες με ≥6 αξιολογήσεις.")
print("Παράδειγμα:", active_users[:5])

# === 2. Δημιουργία πίνακα αξιολογήσεων (user-item matrix) ===
ratings_matrix = df.pivot_table(index='reviewerID', columns='asin', values='overall').fillna(0)

# === 3. Υπολογισμός cosine similarity μεταξύ χρηστών ===
user_similarity = cosine_similarity(ratings_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

# === 4. Επιλογή χρηστών με τουλάχιστον 2 αξιολογήσεις (για RMSE/Precision) ===
active_users = ratings_matrix[ratings_matrix.gt(0).sum(axis=1) >= 2].index.tolist()
print(f"🧮 Βρέθηκαν {len(active_users)} ενεργοί χρήστες με ≥2 αξιολογήσεις.")

if not active_users:
    print("⚠️ Δεν υπάρχουν αρκετοί ενεργοί χρήστες για αξιολόγηση.")
    exit()

# === 5. Συνάρτηση σύστασης προϊόντων για έναν χρήστη ===
def recommend_products(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return pd.DataFrame()

    # Εύρεση παρόμοιων χρηστών με βάση similarity
    sim_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    similar_users_ratings = ratings_matrix.loc[sim_users.index]

    # Σταθμισμένος μέσος όρος αξιολογήσεων
    weighted_ratings = similar_users_ratings.T.dot(sim_users)
    recommendation_scores = weighted_ratings / sim_users.sum()

    # Εύρεση μη αξιολογημένων προϊόντων
    user_rated = ratings_matrix.loc[user_id]
    unrated_items = user_rated[user_rated == 0].index

    # Επιλογή των top προτάσεων
    recommendations = recommendation_scores[unrated_items].sort_values(ascending=False).head(top_n)

    # Προσθήκη τίτλων προϊόντων στις προτάσεις
    titles = df[['asin', 'title']].dropna().drop_duplicates().set_index('asin')
    results = pd.DataFrame({'asin': recommendations.index, 'score': recommendations.values})
    results['title'] = results['asin'].map(titles['title'])

    return results[['asin', 'title', 'score']]

# === 6. Επιλογή τυχαίου χρήστη για προβολή προτάσεων ===
sample_user = random.choice(active_users)
print(f"Επιλεγμένος χρήστης: {sample_user}")
print(recommend_products(sample_user))

# === 7. Υπολογισμός RMSE ===
def evaluate_rmse(sample_users, top_n=5):
    actual = []
    predicted = []

    for user in sample_users:
        if user not in ratings_matrix.index:
            continue

        # Ανάλυση αξιολογήσεων
        user_rated = ratings_matrix.loc[user]
        rated_items = user_rated[user_rated > 0].index.tolist()

        if len(rated_items) < 2:
            continue

        # Διαχωρισμός σε train/test (τελευταίο item για test)
        train_items = rated_items[:-1]
        test_item = rated_items[-1]

        temp_matrix = ratings_matrix.copy()
        temp_matrix.loc[user, test_item] = 0  # "κρύβουμε" το test item

        sim_users = user_similarity_df[user].sort_values(ascending=False)[1:]
        similar_users_ratings = temp_matrix.loc[sim_users.index]
        weighted_ratings = similar_users_ratings.T.dot(sim_users)
        predicted_rating = weighted_ratings[test_item] / sim_users.sum()

        if not np.isnan(predicted_rating):
            actual.append(ratings_matrix.loc[user, test_item])
            predicted.append(predicted_rating)

    if actual:
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print(f"RMSE: {rmse:.4f}")
    else:
        print("Δεν βρέθηκαν κατάλληλοι χρήστες για αξιολόγηση.")

# === 8. Precision@K και Recall@K ===
def evaluate_precision_recall(sample_users, k=5, min_rating=3):
    precision_list = []
    recall_list = []

    for user in sample_users:
        if user not in ratings_matrix.index:
            continue

        user_rated = ratings_matrix.loc[user]
        rated_items = user_rated[user_rated > 0].index.tolist()

        if len(rated_items) < k + 1:
            continue

        test_item = rated_items[-1]
        train_items = rated_items[:-1]

        temp_matrix = ratings_matrix.copy()
        temp_matrix.loc[user, test_item] = 0

        sim_users = user_similarity_df[user].sort_values(ascending=False)[1:]
        similar_users_ratings = temp_matrix.loc[sim_users.index]
        weighted_ratings = similar_users_ratings.T.dot(sim_users)
        predicted_scores = weighted_ratings / sim_users.sum()

        # Top-k προτάσεις
        top_k = predicted_scores.sort_values(ascending=False).head(k).index.tolist()

        # Precision/Recall με βάση το αν περιλαμβάνεται το test item
        relevant_items = [test_item]
        retrieved_relevant = [item for item in top_k if item in relevant_items]

        precision = len(retrieved_relevant) / k
        recall = len(retrieved_relevant) / len(relevant_items)

        precision_list.append(precision)
        recall_list.append(recall)

    if precision_list:
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        print(f"Precision@{k}: {avg_precision:.4f}")
        print(f"Recall@{k}: {avg_recall:.4f}")
    else:
        print("Δεν βρέθηκαν κατάλληλα δεδομένα για Precision/Recall.")

# === 9. Εκτέλεση αξιολόγησης σε δείγμα 20 χρηστών ===
sample = random.sample(active_users, min(20, len(active_users)))
evaluate_rmse(sample)
evaluate_precision_recall(sample, k=5)
