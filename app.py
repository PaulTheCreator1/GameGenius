from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import io
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# === 1. Φόρτωση δεδομένων ===
df = pd.read_csv("video_games_ready.csv")
df = df[df['reviewerID'].notna()]
df = df[df['title'].notna()]
df = df[df['overall'].notna()]
df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text'])
df['tfidf_index'] = range(len(df))

# === 2. Fine-Tuned Hybrid Recommender ===
def hybrid_recommend_for_user(user_id, top_n=10, min_rating=3, weight_user=0.6, weight_content=0.4):
    user_data = df[df['reviewerID'] == user_id]
    liked = user_data[user_data['overall'] >= min_rating]
    if liked.empty:
        return []

    liked_indices = liked['tfidf_index'].tolist()
    user_profile = np.asarray(tfidf_matrix[liked_indices].mean(axis=0))

    sim_user = cosine_similarity(user_profile, tfidf_matrix).flatten()

    sim_content = np.zeros(len(df))
    for i in liked_indices:
        sim_content += cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
    sim_content /= len(liked_indices)

    hybrid_similarity = (weight_user * sim_user) + (weight_content * sim_content)
    df['similarity'] = hybrid_similarity

    seen = user_data['asin'].tolist()
    results = df[~df['asin'].isin(seen)]
    results = results[results['similarity'] > 0]

    return results[['asin', 'title', 'similarity', 'imUrl']] \
        .drop_duplicates(subset='asin').sort_values(by='similarity', ascending=False).head(top_n).to_dict(orient='records')

# === 3. Προτάσεις βάσει χρήστη ===
def recommend_for_user(user_id, top_n=10, min_rating=3):
    user_data = df[df['reviewerID'] == user_id]
    liked = user_data[user_data['overall'] >= min_rating]
    if liked.empty:
        return []

    liked_indices = liked['tfidf_index'].tolist()
    user_profile = np.asarray(tfidf_matrix[liked_indices].mean(axis=0))
    similarities = cosine_similarity(user_profile, tfidf_matrix).flatten()
    df['similarity'] = similarities

    seen = user_data['asin'].tolist()
    results = df[~df['asin'].isin(seen)]
    results = results[results['similarity'] > 0]
    return results[['asin', 'title', 'similarity', 'imUrl']] \
        .drop_duplicates(subset='asin').sort_values(by='similarity', ascending=False).head(top_n).to_dict(orient='records')

# === 4. Προτάσεις βάσει προϊόντος ===
def recommend_for_product(asin, top_n=5):
    product_index = df[df['asin'] == asin]['tfidf_index']
    if product_index.empty:
        return []

    index = product_index.values[0]
    similarities = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    df['similarity'] = similarities

    results = df[df['asin'] != asin].copy()
    results = results[results['similarity'] > 0]
    return results[['asin', 'title', 'similarity', 'imUrl']] \
        .drop_duplicates(subset='asin').sort_values(by='similarity', ascending=False).head(top_n).to_dict(orient='records')

# === 5. Αρχική σελίδα ===
@app.route("/")
def home():
    users = df['reviewerID'].value_counts().head(50).index.tolist()
    return render_template("index.html", users=users, active_tab="user",)

# === 6. Προτάσεις για χρήστη ===
@app.route("/recommendations")
def html_recommendations():
    user_id = request.args.get("user_id", "")
    recs = recommend_for_user(user_id)
    return render_template("recommendations.html", user_id=user_id, recommendations=recs, )

# === 7. Προτάσεις hybrid ===
@app.route("/hybrid", methods=["GET", "POST"])
def hybrid_route():
    if request.method == "POST":
        user_id = request.form.get("user_id", "")
        try:
            weight_user = float(request.form.get("user_weight", 0.6))
            weight_content = float(request.form.get("content_weight", 0.4))
        except:
            weight_user, weight_content = 0.6, 0.4
    else:
        user_id = request.args.get("user_id", "")
        try:
            weight_user = float(request.args.get("user_weight", 0.6))
            weight_content = float(request.args.get("content_weight", 0.4))
        except:
            weight_user, weight_content = 0.6, 0.4

    recs = hybrid_recommend_for_user(
        user_id,
        weight_user=weight_user,
        weight_content=weight_content
    )
    return render_template("recommendations.html", user_id=user_id, recommendations=recs)



# === 8. Επιλογή χρήστη hybrid ===
@app.route("/hybrid_select")
def hybrid_form():
    users = df['reviewerID'].value_counts().head(50).index.tolist()
    return render_template("hybrid.html", users=users, active_tab="hybrid", )

# === 9. Επιλογή προϊόντος ===
@app.route("/product")
def product_form():
    products = df[['asin', 'title']].dropna().drop_duplicates()
    if len(products) > 50:
        products = products.sample(50)
    products = products.to_dict(orient='records')
    return render_template("product.html", products=products, active_tab="product",)

# === 10. Προτάσεις βάσει προϊόντος ===
@app.route("/similar_products")
def similar_products():
    asin = request.args.get("asin")
    recs = recommend_for_product(asin)
    return render_template("recommendations.html", user_id=asin, recommendations=recs, )

# === 11. Dashboard ===
@app.route("/dashboard")
def dashboard():
    top_users = df['reviewerID'].value_counts().head(5)
    top_products = df['title'].value_counts().head(5)
    avg_rating = df['overall'].mean()

    return render_template(
        "dashboard.html",
        top_users=top_users,
        top_products=top_products,
        avg_rating=round(avg_rating, 2),
        active_tab="dashboard",
    )

# === 12. JSON API ===
@app.route("/recommend", methods=["GET"])
def api_recommend():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    recs = recommend_for_user(user_id)
    if not recs:
        return jsonify({"message": f"No good reviews for user '{user_id}'"}), 404
    return jsonify(recs)

@app.route("/about")
def about():
    return render_template("about.html", active_tab="", )

@app.route("/metrics")
def metrics():
    return render_template("evaluation_page.html", active_tab="",)

@app.route("/export_csv")
def export_csv():
    user_id = request.args.get("user_id", "")
    recs = recommend_for_user(user_id)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ASIN", "Title", "Similarity Score"])

    for r in recs:
        writer.writerow([r["asin"], r["title"], f'{r["similarity"]:.4f}'])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        download_name=f"recommendations_{user_id}.csv",
        as_attachment=True
    )
    
@app.route("/favorites")
def favorites():
    return render_template("favorites.html", active_tab="favorites", )

# === 13. Εκκίνηση ===
if __name__ == '__main__':
    app.run(debug=True)
