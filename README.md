# GameGenius – Σύστημα Συστάσεων για Online Καταστήματα

Το **GameGenius** είναι μια ολοκληρωμένη web εφαρμογή συστάσεων που βασίζεται σε Flask και Python. Υποστηρίζει τρεις τύπους recommendation αλγορίθμων (User-based, Content-based, Hybrid) και διαθέτει πλήρες responsive περιβάλλον με dark/light mode, sliders βαρών και dashboard στατιστικών. Η εφαρμογή αποτελεί μέρος πτυχιακής εργασίας στο Metropolitan College - University of East London

## Δομή Αρχείων

GameGenius/
├── app.py
├── /templates/
│   ├── index.html
│   ├── product.html
│   ├── hybrid.html
│   ├── dashboard.html
│   ├── favorites.html
│   ├── about.html
│   ├── evaluation_page.html
│   ├── recommendations.html
│   └── base.html
├── /static/
│   ├── logo.png
│   ├── rmse_comparison.png
│   ├── rmse_report.pdf
│   ├── video_games_ready.json
│   └── style.css
├── content_based_user_recommender.py
├── content_metrics.py
├── dummy_users.csv
├── generate_dummy_users.py
├── hybrid_metrics.py
├── make_ready_json.py
├── merge_users.py
├── plot_final_metrics.py
├── prepare_video_games_dataset.py
├── user_based_recommender.py
├── user_metrics.py
├── video_games_ready.csv
├── video_games_ready_with_dummy.csv
└── README.md

## Προαπαιτούμενα

- Python 3.8+
- pip (Python package manager)
- Συνιστάται εικονικό περιβάλλον (virtualenv)

## Εγκατάσταση

1. Κατέβασε ή κάνε clone το αποθετήριο:

```
git clone https://github.com/PaulTheCreator1/GameGenius.git
cd GameGenius
```

2. Δημιούργησε εικονικό περιβάλλον (προαιρετικά):

```
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate         # Windows
```

3. Εγκατάσταση απαιτούμενων βιβλιοθηκών:

```
pip install -r requirements.txt
```

## Εκκίνηση Εφαρμογής

```bash
python app.py
```

Άνοιξε τον browser στη διεύθυνση: http://127.0.0.1:5000

## Βασικά Χαρακτηριστικά

- 3 Recommendation Αλγόριθμοι: User-based, Content-based, Hybrid
- Δυναμική επιλογή βαρών στο Hybrid μέσω sliders
- Υποστήριξη dark/light mode
- Αποθήκευση αγαπημένων με localStorage
- Εμφάνιση στατιστικών: Top χρήστες, Top προϊόντα, Μέση βαθμολογία
- Υπολογισμός RMSE, Precision@5, Recall@5 για κάθε μοντέλο

## Ενσωματωμένοι Αλγόριθμοι

| Αλγόριθμος        | Περιγραφή |
|------------------|-----------|
| User-based       | Σύγκριση χρηστών με βάση αξιολογήσεις (cosine similarity) |
| Content-based    | Σύγκριση προϊόντων με βάση περιγραφή/tags (TF-IDF + Cosine) |
| Hybrid           | Συνδυασμός user & content με βαρών από sliders |

## Δεδομένα

- Dataset: Amazon Video Games Reviews
- Προεπεξεργασμένο για χρήση (CSV/JSON)
- Φιλτραρισμένα δεδομένα για >= 6 αξιολογήσεις/χρήστη
- Πάνω από 38.000 γραμμές χρήσιμες για προτάσεις

## Τεστ Περιβάλλοντος

- Python 3.9.6
- Windows 10 / Ubuntu 22.04
- Google Chrome / Edge
- Flask 2.3.x

## requirements

Flask  
pandas  
scikit-learn  
matplotlib

##  Συντάκτης

Στεφανίδης Παύλος  
BSc (Hons) Computer Science  
Metropolitan College - University of East London
Ιούνιος 2025

## Άδεια Χρήσης

Το project διατίθεται αποκλειστικά για εκπαιδευτική χρήση. Απαγορεύεται η αναδιανομή ή εμπορική χρήση χωρίς σχετική άδεια.
