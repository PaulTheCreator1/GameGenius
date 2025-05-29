import matplotlib.pyplot as plt
import numpy as np

# Ορισμός των μοντέλων προς σύγκριση
models = ['User-based', 'Content-based', 'Hybrid']

# Τιμές μετρικών για κάθε μοντέλο
rmse = [2.7434, 0.0, 2.9732]
precision = [0.1600, 0.0, 0.7000]
recall = [0.8000, 0.0, 0.7000]

# Θέσεις στον άξονα Χ για τις μπάρες
x = np.arange(len(models))
width = 0.25  # πλάτος κάθε μπάρας

# Δημιουργία του γραφήματος και άξονα
fig, ax = plt.subplots(figsize=(10, 6))

# Σχεδίαση μπάρας RMSE
ax.bar(x - width, rmse, width, label='RMSE', color='steelblue')

# Σχεδίαση μπάρας Precision@5
ax.bar(x, precision, width, label='Precision@5', color='darkorange')

# Σχεδίαση μπάρας Recall@5
ax.bar(x + width, recall, width, label='Recall@5', color='seagreen')

# Ρυθμίσεις άξονα
ax.set_ylabel('Τιμή')
ax.set_xlabel('Μοντέλο')
ax.set_title('Σύγκριση Μετρικών Ανά Μοντέλο')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Προσθήκη αριθμητικών τιμών πάνω από τις μπάρες
for i in range(len(models)):
    ax.text(x[i] - width, rmse[i] + 0.1, f"{rmse[i]:.2f}", ha='center')
    ax.text(x[i], precision[i] + 0.05, f"{precision[i]:.2f}", ha='center')
    ax.text(x[i] + width, recall[i] + 0.05, f"{recall[i]:.2f}", ha='center')

# Τελική διάταξη και αποθήκευση ως εικόνα
plt.tight_layout()
plt.savefig("static/rmse_comparison.png")
print("Νέο γράφημα αποθηκεύτηκε: static/rmse_comparison.png")
