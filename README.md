# Hybrid DistilBERT Phishing Detection

A feature-level hybrid model for automated phishing email detection using DistilBERT and structural URL features.

##  Overview
Phishing attacks are evolving. Traditional methods (blacklists, regex, or text-only ML) often miss sophisticated attacks. This project presents a **Hybrid DistilBERT Architecture** that fuses deep semantic text analysis with URL structural signals to achieve **99.1% classification accuracy**.

##  Key Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 94.0% | 95% | 95% | 95% |
| Random Forest | 95.0% | 96% | 95% | 96% |
| SVM | 97.0% | 98% | 96% | 97% |
| **Hybrid DistilBERT** | **99.1%** | **98%** | **99%** | **98%** |

##  Architecture
- **Text Modality:** DistilBERT (768-dim [CLS] embeddings).
- **Structural Modality:** Handcrafted URL features (binary encoding for suspicious patterns).
- **Fusion:** Feature-level concatenation (769-dim) fed into a dense classifier with Dropout (0.3) to prevent overfitting.

##  Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run the model training: `python hybrid_model.py`
3. Generate results visualizations: `python generate_visuals.py`

##  Publication
Published in *Journal of Computing & Biomedical Informatics*, Vol. 9, Issue 1, 2025.

##  Visualizations
See uploaded images: `confusion_matrix.png` & `performance_graph.png`
