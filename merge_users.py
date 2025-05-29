import pandas as pd

# === 1. Φόρτωση αρχικών δεδομένων ===
# Το αρχικό dataset με τις κανονικές αξιολογήσεις
original = pd.read_csv("video_games_ready.csv")

# Το dummy dataset με τεχνητούς χρήστες για αξιολόγηση Precision/Recall
dummy = pd.read_csv("dummy_users.csv")

# === 2. Αντιστοίχιση στηλών ===
# Ελέγχει αν υπάρχουν στήλες στο original που δεν υπάρχουν στο dummy και τις προσθέτει
for col in original.columns:
    if col not in dummy.columns:
        dummy[col] = ""  # Προσθήκη με κενές τιμές

# Αναδιάταξη των στηλών του dummy για να ταιριάζει με το original
dummy = dummy[original.columns]

# === 3. Συγχώνευση των δύο DataFrames ===
# Συγχωνεύει σε νέο αρχείο — χωρίς να χαθούν στήλες
merged = pd.concat([original, dummy], ignore_index=True)

# === 4. Αποθήκευση του τελικού dataset ===
merged.to_csv("video_games_ready_with_dummy.csv", index=False)

print("✅ Συγχωνεύτηκαν σε video_games_ready_with_dummy.csv")
