import matplotlib.pyplot as plt
import numpy as np

# Ορισμός μοντέλων που συγκρίνονται
models = ['User-based', 'Content-based', 'Hybrid']

# Τιμές των μετρικών για κάθε μοντέλο
rmse = [2.7434, 0.0, 0.0]
precision = [0.1600, 0.0, 0.0]
recall = [0.8000, 0.0, 0.0]

# Δημιουργία άξονα Χ (θέσεις για τα bars)
x = np.arange(len(models))
width = 0.25  # πλάτος κάθε μπάρας

# Δημιουργία καμβά (figure) και άξονα (axis)
fig, ax = plt.subplots(figsize=(10, 6))

# Προσθήκη μπλε μπάρας για RMSE
ax.bar(x - width, rmse, width, label='RMSE', color='steelblue')

# Πορτοκαλί μπάρα για Precision@5
ax.bar(x, precision, width, label='Precision@5', color='darkorange')

# Πράσινη μπάρα για Recall@5
ax.bar(x + width, recall, width, label='Recall@5', color='seagreen')

# Ετικέτες και τίτλοι στο γράφημα
ax.set_ylabel('Τιμή')
ax.set_xlabel('Μοντέλο')
ax.set_title('Σύγκριση Μετρικών Ανά Μοντέλο')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Προσθήκη αριθμητικής τιμής πάνω από κάθε μπάρα
for i in range(len(models)):
    ax.text(x[i] - width, rmse[i] + 0.05, f"{rmse[i]:.2f}", ha='center')
    ax.text(x[i], precision[i] + 0.05, f"{precision[i]:.2f}", ha='center')
    ax.text(x[i] + width, recall[i] + 0.05, f"{recall[i]:.2f}", ha='center')

# Βελτιστοποίηση layout και αποθήκευση ως εικόνα PNG
plt.tight_layout()
plt.savefig("static/rmse_comparison.png")
print("Grouped γράφημα αποθηκεύτηκε: static/rmse_comparison.png")
