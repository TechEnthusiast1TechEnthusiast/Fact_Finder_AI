
Fake News Detection System
A machine learning-based system to detect and classify news articles as "fake" or "real" using various classification models such as Logistic Regression, Decision Tree, and Random Forest. The project involves extensive text preprocessing, feature extraction, and data visualization techniques to achieve high accuracy in identifying fake news.

Overview
The Fake News Detection System is designed to classify news articles as either "fake" or "real" using machine learning techniques. The system preprocesses text data, vectorizes the content using TF-IDF, and applies various classifiers to achieve high accuracy.

Key Features:
Text preprocessing: Removal of stopwords, punctuation, and conversion to lowercase.
Feature extraction: Term frequency-inverse document frequency (TF-IDF) vectorization.
Multiple machine learning models: Logistic Regression, Decision Tree, and Random Forest.
Data visualization: Word clouds, frequent word analysis, and confusion matrices to assess model performance.
Dataset
The dataset consists of two parts:

Fake News: A collection of 23,481 fake news articles.
Real News: A collection of 21,417 real news articles.
The dataset includes the following columns:

title: The title of the news article.
text: The content of the news article.
subject: The subject/category of the article.
target: Labels denoting whether the article is real or fake.
Installation
To get started with the project, follow the instructions below:

Prerequisites
Python 3.x
Jupyter Notebook (optional but recommended)
Libraries: pandas, numpy, sklearn, seaborn, matplotlib, nltk, wordcloud



The requirements.txt file should include:
pandas
numpy
scikit-learn
seaborn
matplotlib
nltk
wordcloud


Usage
Data Preprocessing: The text data undergoes various preprocessing steps like lowercase conversion, punctuation removal, and stopword removal.
Model Training: Train the model using train_test_split to split the dataset into training and test sets.
Classification: Use Logistic Regression, Decision Tree, and Random Forest models to classify the news articles.
Evaluation: Visualize the model performance using confusion matrices and analyze the accuracy.
Run the Jupyter notebook or the Python scripts to see the outputs, such as word clouds, frequent word analysis, and model accuracy.

Models and Results
Logistic Regression: Achieved an accuracy of 99.01%.
Decision Tree: Achieved an accuracy of 99.59%.
Random Forest: Achieved an accuracy of 99.23%.
All models are evaluated using confusion matrices to assess their classification accuracy.

Features
Text Preprocessing: Applied techniques such as stopword removal, punctuation removal, and TF-IDF vectorization.
Model Evaluation: Plotted confusion matrices and bar charts for visualizing data distribution and model accuracy.
Word Cloud Generation: Created word clouds for both fake and real news articles for better insights.
Frequent Word Analysis: Visualized the most common words used in fake and real news articles.
Technologies Used
Python: Programming language for implementing the system.
Pandas & Numpy: Data manipulation and analysis.
NLTK: Natural language processing for text data cleaning.
Scikit-learn: Machine learning library used for classification and model evaluation.
Matplotlib & Seaborn: Data visualization libraries.
WordCloud: Library to generate word clouds.
Contributing
Contributions are welcome! If you have suggestions or would like to contribute, please submit an issue or pull request.

This project is licensed under the MIT License. See the LICENSE file for details.
