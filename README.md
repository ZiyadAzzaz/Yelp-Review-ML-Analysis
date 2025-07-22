# 🧠 Yelp Review ML Analysis
> Predicting business review ratings using text classification and sentiment analysis with Python and machine learning.

![Platform](https://img.shields.io/badge/Platform-Jupyter--Notebook-blue)
![Language](https://img.shields.io/badge/Language-Python-green)
![Model](https://img.shields.io/badge/Model-LogisticRegression-purple)
![Dataset](https://img.shields.io/badge/Dataset-Yelp%20Reviews-red)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 📚 Table of Contents
- [📌 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🧰 Tech Stack](#-tech-stack)
- [⚙️ ML Pipeline](#️-ml-pipeline)
- [🔧 Feature Engineering](#-feature-engineering)
- [📈 Sample Code Snippets](#-sample-code-snippets)
- [📊 Evaluation Results](#-evaluation-results)
- [🤖 Model Comparison](#-model-comparison)
- [📁 Project Structure](#-project-structure)
- [📊 Visualizations](#-visualizations)
- [📈 Business Insights](#-business-insights)
- [🌍 Real-World Use Cases](#-real-world-use-cases)
- [📊 Model Discussion](#-model-discussion)
- [🛠️ How to Run](#-how-to-run)
- [🎯 Learning Objectives](#-learning-objectives)
- [📄 License](#-license)
- [🙋‍♂️ Author](#-author)

---

## 📌 Project Overview

This project focuses on analyzing the **Yelp business reviews dataset** and building a machine learning pipeline to:
- Explore the relationship between review text and business star ratings.
- Extract text features using TF-IDF vectorization.
- Train a **Logistic Regression** model for rating prediction.
- Evaluate performance with classification metrics and cross-validation.

The entire analysis is framed in a business context to support customer satisfaction insights, rating reliability, and feedback value.


👉 **[View the Full Notebook Here](https://github.com/ZiyadAzzaz/Yelp-Review-ML-Analysis/blob/main/ML.ipynb)**

---

## 📊 Dataset

The dataset (`yelp.csv`) includes:
- `text`: Raw review content (used for NLP)
- `stars`: Target variable (rating from 1 to 5 stars)
- Other features like `cool`, `useful`, `funny` (can be explored for advanced models)

---

## 🧰 Tech Stack

- 💻 **Language**: Python  
- 📒 **Platform**: Jupyter Notebook  
- 📚 **Libraries**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`

---

## ⚙️ ML Pipeline

1. **Text Vectorization**
   - Used `TfidfVectorizer(max_features=5000)` on the review text column.
2. **Model Building**
   - Trained a `LogisticRegression(max_iter=1000)` classifier.
3. **Model Evaluation**
   - Used `train_test_split` with 80/20 ratio
   - Evaluated with accuracy and classification report
   - Used `cross_val_score` with 5-fold cross-validation

---

## 🔧 Feature Engineering

- Cleaned text column (removed nulls, punctuation, stopwords)
- Applied **TF-IDF vectorization**
- Merged with numeric columns like `cool`, `useful` (optional extended version)
- Reduced dimensionality by limiting TF-IDF features

---

## 📈 Sample Code Snippets

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df['text'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_text, df['stars'], test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---

## 📊 Evaluation Results
This Report After we make our starts into 2 Classes
```
Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.62      0.71       597
           1       0.85      0.95      0.90      1403

    accuracy                           0.85      2000
   macro avg       0.85      0.78      0.81      2000
weighted avg       0.85      0.85      0.84      2000
```

---

## 🤖 Model Comparison
Before Convert To Classes

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | 54%      |
| Random Forest         |47%      |


> Logistic Regression offered the best performance with minimal training time.
**After Binary Classification Accuracy Of Logistic Regression Will be 85%**

---

## 📁 Project Structure

```
Yelp-Review-ML-Analysis/
├── yelp.csv
├── Yelp_Review_Analysis.ipynb
├── README.md
```

---

## 📊 Visualizations

- 📌 WordCloud of top tokens by star rating
- 📌 Bar plot of star rating distribution
- 📌 Heatmap of TF-IDF importance
- 📌 Confusion matrix of predictions
- 📌 Accuracy plot across folds (cross-validation)

---

## 📈 Business Insights

- Positive (4–5 stars) reviews are more consistent in sentiment and structure.
- Keywords such as “excellent”, “fast”, “friendly” strongly align with higher ratings.
- Negative reviews are more diverse and include emotion-charged terms.

---

## 🌍 Real-World Use Cases

- 🏷️ Auto-tagging reviews for moderation or support prioritization
- ⭐ Rating prediction when stars are missing or unclear
- 📈 Tracking customer experience over time
- 📬 Training sentiment-aware recommendation systems

---

## 📊 Model Discussion

The model achieved around 85% accuracy, which is reasonable for multi-class text classification.
Slight confusion occurred between classes 2–3, which is expected due to overlapping sentiment language.

Improvements could include:
- Balancing classes using resampling
- Exploring additional NLP models like SVM or Gradient Boosting
- Using pretrained embeddings or transformer-based models

---

## 🛠️ How to Run

```bash
git clone https://github.com/ZiyadAzzaz/Yelp-Review-ML-Analysis.git
cd Yelp-Review-ML-Analysis
jupyter notebook Yelp_Review_Analysis.ipynb
```

---

## 🎯 Learning Objectives

- Understand and implement TF-IDF feature extraction
- Use Logistic Regression for text classification
- Evaluate models with cross-validation and interpret metrics
- Gain insights from NLP in a business context

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

Developed by Ziyad Azzaz  
GitHub: https://github.com/ZiyadAzzaz
