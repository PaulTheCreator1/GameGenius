import pandas as pd
import random

# Φόρτωσε το αρχικό dataset
df = pd.read_csv("video_games_ready.csv")

# Πάρε μοναδικά προϊόντα με ASIN & τίτλο
products = df[['asin', 'title']].drop_duplicates().reset_index(drop=True)

# Δημιούργησε 10 χρήστες με ≥ 6 αξιολογήσεις
dummy_reviews = []

for i in range(1, 11):  # User1 - User10
    user_id = f"User{i}"
    sampled = products.sample(n=8, random_state=i)  # 8 αξιολογήσεις
    for _, row in sampled.iterrows():
        dummy_reviews.append({
            'reviewerID': user_id,
            'asin': row['asin'],
            'overall': random.randint(1, 5),
            'title': row['title']
        })

# Αποθήκευσε νέο αρχείο
dummy_df = pd.DataFrame(dummy_reviews)
dummy_df.to_csv("dummy_users.csv", index=False)

print("Δημιουργήθηκαν 10 dummy χρήστες με 8 αξιολογήσεις έκαστος.")
