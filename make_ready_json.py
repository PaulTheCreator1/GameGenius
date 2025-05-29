import pandas as pd

# === 1. Φόρτωση του dataset CSV ===
df = pd.read_csv("video_games_ready.csv")

# === 2. Επιλογή βασικών στηλών για το frontend ===
# Κρατάμε μόνο ASIN (ID προϊόντος), τίτλο και URL εικόνας
df = df[["asin", "title", "imUrl"]].dropna()

# === 3. Αφαίρεση διπλότυπων εγγραφών με βάση το asin ===
# Έτσι δεν θα εμφανίζονται επαναλαμβανόμενα προϊόντα στο interface
df = df.drop_duplicates(subset="asin")

# === 4. Αποθήκευση σε JSON για χρήση από την web διεπαφή ===
df.to_json("static/video_games_ready.json", orient="records", indent=2)
