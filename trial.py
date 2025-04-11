import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer



#load true news daatset
true_df = pd.read_csv(r'C:\trial fake news\News _dataset\True.csv')
true_df['label'] = 0 
#load fake news daatset
fake_df = pd.read_csv(r'C:\trial fake news\News _dataset\Fake.csv')
fake_df['label'] = 1


df = pd.concat([true_df,fake_df],ignore_index=True)


X_train,X_test,y_train,y_test = train_test_split(df['text'],df['label'],test_size=0.2,random_state=42)


#train the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

#make prediction
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)


user_input = input("Enter a news article: ")

# Make a prediction
prediction = model.predict([user_input])

# Display the result
if prediction[0] == 0:
    print("The news is likely to be true.")
else:
    print("The news is likely to be fake.")