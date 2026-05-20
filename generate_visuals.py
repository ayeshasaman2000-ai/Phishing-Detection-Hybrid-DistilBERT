import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create results folder
os.makedirs('results', exist_ok=True)

# --- 1. Confusion Matrix ---
confusion_matrix = np.array([[39250, 345],
                             [429, 42462]])
labels = ['Legitimate', 'Phishing']

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Hybrid DistilBERT: Confusion Matrix\n(Test Set)', fontsize=16)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300)
print("✅ Saved: results/confusion_matrix.png")

# --- 2. Performance Graph ---
models = ['Logistic Regression', 'Random Forest', 'SVM', 'Hybrid DistilBERT']
accuracies = [94.0, 95.0, 97.0, 99.1]
colors = ['#6c757d', '#6c757d', '#6c757d', '#007bff']

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=colors)
plt.title('Model Performance Comparison: Accuracy (%)', fontsize=16)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(90, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f'{yval}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/performance_graph.png', dpi=300)
print("✅ Saved: results/performance_graph.png")
