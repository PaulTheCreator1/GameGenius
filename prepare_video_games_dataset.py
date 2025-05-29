import json
import pandas as pd

# === Ρυθμίσεις αρχείων ===
reviews_path = "reviews_Video_Games.json"       # Αρχείο με αξιολογήσεις χρηστών
meta_path = "meta_Video_Games.json"             # Αρχείο με πληροφορίες προϊόντων (τίτλος, περιγραφή, τιμή)
output_path = "video_games_ready.csv"           # Τελικό όνομα αρχείου εξόδου
sample_size = 10000                             # Μέγιστος αριθμός reviews για φόρτωση

# === 1. Φόρτωση πρώτων 10.000 reviews ===
print("Φόρτωση reviews...")
reviews = []

# Ανάγνωση γραμμή-γραμμή του αρχείου JSON
with open(reviews_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= sample_size:
            break
        reviews.append(json.loads(line.strip()))

# Μετατροπή σε DataFrame
df_reviews = pd.DataFrame(reviews)
print(f"Φορτώθηκαν {len(df_reviews)} reviews")
print("Στήλες reviews:", df_reviews.columns.tolist())

# === 2. Φόρτωση metadata (προϊόντα) ===
print("\n Φόρτωση metadata...")
meta = []

with open(meta_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
    for line in f:
        try:
            # Κάποιες εγγραφές έχουν μονά εισαγωγικά αντί για διπλά, διορθώνονται
            obj = json.loads(line.strip().replace("'", '"'))
            if 'asin' in obj:  # Αν έχει αναγνωριστικό προϊόντος
                meta.append(obj)
        except Exception:
            continue  # Αγνόηση ελαττωματικών γραμμών

# Μετατροπή σε DataFrame
df_meta = pd.DataFrame(meta)
print(f"Φορτώθηκαν {len(df_meta)} metadata entries")
print("Στήλες metadata:", df_meta.columns.tolist())

# === 3. Καθαρισμός και συγχώνευση ===
print("\n Καθαρισμός και συγχώνευση...")

# Αφαίρεση εγγραφών χωρίς ASIN
df_reviews = df_reviews[df_reviews['asin'].notna()]
df_meta = df_meta[df_meta['asin'].notna()]

# Εσωτερική ένωση των δύο πινάκων πάνω στο πεδίο 'asin'
df = pd.merge(df_reviews, df_meta, on='asin', how='inner')

# Κρατάμε μόνο τα βασικά πεδία που χρειαζόμαστε
required_columns = ['reviewerID', 'asin', 'overall', 'title', 'description', 'brand', 'price', 'imUrl']
df = df[[col for col in required_columns if col in df.columns]]

# Αφαίρεση εγγραφών με κενές τιμές σε βασικά πεδία
df = df[df['title'].notna()]
df = df[df['overall'].notna()]
df = df[df['reviewerID'].notna()]

print(f"Τελικό dataset: {len(df)} εγγραφές")
print("Τελικές στήλες:", df.columns.tolist())

# === 4. Αποθήκευση σε CSV ===
df.to_csv(output_path, index=False)
print(f"\n Αποθηκεύτηκε ως: {output_path}")
