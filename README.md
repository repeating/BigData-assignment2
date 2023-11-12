# Twitter-Sentiment-Analysis-BigData

## Project Overview
This project, undertaken by Fadi Younes, Artur Akhmetshin, and Dinar Zayahov, focuses on the real-time analysis of Twitter data. The primary goal is to fetch tweets from a stream, preprocess them, and classify these tweets based on their sentiment.

## Features
- **Stream Processing**: The application processes live Twitter streams and captures tweets for analysis.
- **Preprocessing**: Implements steps like removing aliases, links, punctuation, stop words, and converting text to lowercase.
- **Sentiment Analysis**: Classifies tweets into different sentiment categories.
- **Model Comparison**: Compares various machine learning models for optimal classification accuracy.

## Results
The project tested several machine learning algorithms, including Linear SVM, Random Forest Classifier, Naive Bayes Classifier, and Logistic Regression. The LinearSVC with CountVectorizer was identified as the best model, achieving the highest F1 score. Detailed accuracy metrics for each model are available in the project report.

## Installation

### Prerequisites
- Scala
- Apache Spark
- An active Twitter developer account for accessing Twitter API

### Steps
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/repeating/Twitter-Sentiment-Analysis-BigData
   ```
2. **Navigate to the Project Directory**:
   ```sh
   cd Twitter-Sentiment-Analysis-BigData
   ```
3. **Install Dependencies**:
   - Ensure Scala and Apache Spark are installed.
   - Install any additional required Scala libraries.

### Configuration
- Set up Twitter API keys in the configuration file.

## Running the Application
1. **Start the Stream**:
   - Execute the script to start fetching tweets.
2. **Run Preprocessing**:
   - Process the raw tweets to a suitable format.
3. **Execute the Classifier**:
   - Run the classifier to categorize tweets based on sentiment.

## Team Responsibilities
- **Artur Akhmetshin**: Implemented the word counter and tweet stream, collected 24-hour data.
- **Dinar Zayahov**: Worked on classification algorithms, and ran wordCount and classifier on a cluster.
- **Fadi Younes**: Implemented preprocessing, worked with classifiers, and tested them.
