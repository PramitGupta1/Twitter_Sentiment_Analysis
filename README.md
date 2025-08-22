# Twitter_Sentiment_Analysis
Machine learning project for Twitter sentiment analysis. Tweets are cleaned, preprocessed, and classified into Positive, Negative, or Neutral using models like Naive Bayes, Logistic Regression, and Linear SVM. Includes real-time &amp; batch predictions with saved model and vectorizer.

**COMPANY**: Star App Solutions
**Job Role**:Intern
**Name**: Pramit Kumar Gupta
**Domain**:Data Analytics
**Duration**:6 Month

# Summary of the project
The objective of this project is to build a machine learning model that can automatically classify tweets into three categories: Positive, Negative, or Neutral. Such classification helps businesses, governments, and researchers understand public opinion on topics, brands, or events in real time, making it a valuable tool for decision-making and strategy development.

The data for this project was sourced from the twitter_training.csv dataset, with an optional reference to the Sentiment140 dataset containing 1.6 million tweets. After preprocessing, the final dataset consisted of cleaned and filtered tweets with sentiment labels. Preprocessing steps included removing URLs, mentions, hashtags, numbers, and special characters, converting text to lowercase, removing stopwords, and applying tokenization and lemmatization. This ensured consistent and normalized text, restricted to only three sentiment categories: Positive, Negative, and Neutral.

Exploratory Data Analysis (EDA) was conducted to better understand the dataset. This involved checking data size, identifying and handling null values or duplicates, reviewing class distribution for balance, and examining sample tweets from each category. EDA ensured that the dataset was reliable for model training and evaluation.

In the model development stage, TF-IDF vectorization was used to convert text into numerical features suitable for machine learning algorithms. Three models were trained and tested: Naive Bayes, Logistic Regression, and Linear Support Vector Machine (SVM). The models were evaluated using standard metrics such as accuracy, precision, recall, F1-score, and confusion matrix. Among these, Linear SVM achieved the best performance, showing high accuracy in distinguishing Positive and Negative tweets, though some confusion remained in the Neutral category.

Deployment and prediction features were also integrated into the project. A real-time prediction function was developed for classifying new tweets instantly, while a batch mode was created to process and score sentiments from CSV files. The trained model and vectorizer were saved, enabling reuse and scalability for future applications.

The project was implemented using Python, with libraries like Pandas and NumPy for data handling, NLTK for preprocessing, and Scikit-learn for model building and evaluation. Visualization was supported by Matplotlib and Seaborn. Deliverables include the cleaned dataset, trained model files, Jupyter Notebook with complete code, and demo predictions in both real-time and batch modes.

In conclusion, this project demonstrates how raw Twitter data can be transformed into meaningful sentiment insights using machine learning. It serves as a strong foundation for real-world applications such as brand monitoring, political analysis, and product feedback mining.

## Output

<img width="1077" height="737" alt="Image" src="https://github.com/user-attachments/assets/a2f1621f-36f0-4e98-8fb9-c16ded7ea3a1" />

<img width="854" height="715" alt="Image" src="https://github.com/user-attachments/assets/1cd8dddf-2534-41f4-87ef-1cb51f28d636" />

<img width="1459" height="745" alt="Image" src="https://github.com/user-attachments/assets/525b4d44-00e1-4475-9a06-f65abf9f7687" />
