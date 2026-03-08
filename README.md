# Song Genre Classification

This repository demonstrates how to build a Natural Language Processing (NLP) pipeline to classify song genres based on lyrics using both traditional machine learning models and transformer-based models.

The project includes model training, evaluation, comparison between models, and a web demo application for interactive predictions.

## Overview

This project explores how machine learning models can understand song lyrics and classify them into genres.

Three models are implemented:

1. **Logistic Regression** - Traditional ML model using TF-IDF features
2. **Random Forest** - Ensemble method with TF-IDF features  
3. **BERT** - Transformer-based model fine-tuned for classification

The workflow includes:

* Data preprocessing and cleaning
* Feature extraction using TF-IDF and BERT embeddings
* Model training and evaluation
* Comparison of model performance
* Web demo application for interactive predictions

## Repository Structure

```
├── hust, song classification using bert_07_03
│   └── btap.ipynb      # Model training and evaluation
│
├── models
│   ├── bert_song_classifier.pth    # Trained BERT model
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── model.py                         # BERT classifier architecture
│
├── web_demo.py                      # Streamlit web demo
│
├── requirements.txt                 # Python dependencies
│
└── README.md
```

## Dataset

The dataset consists of song lyrics labeled by genre.

Example structure:

| Genre  | Lyrics                       | Genre_Index |
|--------|------------------------------|-------------|
| Rock   | I wanna rock tonight...      | 3           |
| Hip-Hop| I got the rhythm and the beat| 1           |

Genres used in the project:

* 0 -> Country
* 1 -> Hip-Hop
* 2 -> Electronic
* 3 -> Rock

Models

The project compares three different approaches.

1 BERT (Transformer Model)

* Pretrained model: bert-base-uncased
* Fine-tuned for genre classification
* Input: tokenized lyrics
* Output: genre prediction

Advantages:

* understands contextual meaning of lyrics
* best performance among tested models

2 Logistic Regression + TF-IDF

Traditional NLP pipeline: Lyrics → TF-IDF Vectorizer → Logistic Regression → Prediction

Advantages:

* fast training
* good baseline for comparison

3 Random Forest + TF-IDF

Ensemble model applied to TF-IDF vectors.

Advantages:

* captures nonlinear relationships
* interpretable baseline model

4 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example results:

| Model                | Accuracy | Macro F1 |
|----------------------|----------|----------|
| Logistic Regression  | ~0.69    | ~0.66    |
| Random Forest        | ~0.64    | ~0.62    |
| BERT                 | 0.74     | 0.72     |

The transformer model achieved the best performance.

5 Web Demo

The repository includes a simple Streamlit web application that allows users to enter song lyrics and select a model for prediction.

Features:

* Input song lyrics
* Select model: BERT, Logistic Regression, Random Forest
* Display predicted genre

Example interface:

Song Genre Classification

Enter lyrics:
[ textbox ]

Select model:
(BERT / Logistic Regression / Random Forest)

[ Predict ]

Predicted Genre: Hip-Hop

Run Web Demo

* Start the Streamlit application:streamlit run web_demo.py
* Then open: http://localhost:8501

Future Improvements:

Possible future extensions:

* Train additional transformer models (DistilBERT, RoBERTa)
* Hyperparameter tuning
* Larger lyrics dataset
* Model deployment using Docker or cloud platforms

Disclaimer

* This project is for educational and research purposes only.
* The models and dataset are used to demonstrate NLP techniques for text classification.

Author

* Developed as part of a Natural Language Processing course project.
