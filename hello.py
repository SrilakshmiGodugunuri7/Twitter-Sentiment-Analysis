from flask import Flask,render_template,request
app = Flask(__name__)

@app.route('/')
def hello_world():
   return render_template("login.html")

@app.route('/home',methods=['POST','GET'])
def home():
   name=request.form['nm']
   return render_template("home.html",name=name)


@app.route('/tweet',methods=['POST','GET'])
def tweet():
      #text classification
   import numpy as np
   import re
   import pickle
   import nltk
   from nltk.corpus import stopwords
   from sklearn.datasets import load_files#for importing the datasets
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.feature_extraction.text import TfidfTransformer
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import confusion_matrix

   #loading the reviews
   reviews=load_files('txt_sentoken/')
   x,y=reviews.data,reviews.target

   #storing as pickle files
   with open('x.pickle','wb') as f:
       pickle.dump(x,f)
   with open('y.pickle','wb') as f:
       pickle.dump(y,f)

   #unpacking the datasets
   with open('x.pickle','rb') as f:
       x=pickle.load(f)
   with open('y.pickle','rb') as f:
       y=pickle.load(f)


   #Creating the corpus
   corpus=[]
   for i in range(len(x)):
       review=re.sub(r'\W',' ',str(x[i]))
       review=review.lower()
       review=re.sub(r'\s+[a-z]\s+',' ',review)
       review=re.sub(r'^[a-z]\s+',' ',review)
       review=re.sub(r'\s+',' ',review)
       corpus.append(review)

   #BOW converting
   vectorizer=CountVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
   x=vectorizer.fit_transform(corpus)
   #print(x[0])

   #converting the BOW to TF-IDF model
   transformer=TfidfTransformer()
   x=transformer.fit_transform(x).toarray()

   #converting TF-IDF vector
   vectorizer=TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
   x=vectorizer.fit_transform(corpus)


   #creating training and testing datasets
   text_train,text_test,sent_train,sent_test=train_test_split(x,y,test_size=0.2,random_state=0)

   #Using the binary classifier- LogisticRegression-training the data
   classifier=LogisticRegression(solver='lbfgs')
   classifier.fit(text_train,sent_train)

   #testing the test data
   sent_pred=classifier.predict(text_test)
   cm=confusion_matrix(sent_test,sent_pred)

   #pickling the classifier
   with open('classifier.pickle','wb') as f:
       pickle.dump(classifier,f)

   #pickling the vectorizer
   with open('model.pickle','wb') as f:
       pickle.dump(vectorizer,f)

   #Unpickling the classifier and vectorizer
   with open('classifier.pickle','rb') as f:
       clf=pickle.load(f)

   with open('model.pickle','rb') as f:
       tfidf=pickle.load(f)

   sample=[]
   sample.append(request.form['tweet'])
   sample=tfidf.transform(sample).toarray()
   result=clf.predict(sample)
   if result[0]==1:
               return render_template("tweet.html",result="Positive")
   else:
               return render_template("tweet.html",result="Negative")

@app.route('/get_tweet',methods=['POST','GET'])
def get_tweet():
   import tweepy
   import pickle
   import re
   import nltk
   from nltk.corpus import stopwords
   from tweepy import OAuthHandler
   import heapq
   import sys
   import numpy as np
   import matplotlib.pyplot as plt


   #sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)

   #Initializing the keys
   consumer_key='cxo42N8TgBtLVPiomGt203E36'
   consumer_secret='WY1eNWkNzJ5b5ZENuJbgRgo3U2Y0IkisCxptoIHd96DxTDojH2'
   access_token='835087655587168257-rGzf4FZFEafctvd458hsAWc6KnmbWe5'
   access_secret='tqnpfZOCIKqAn5bfDp78FwzZlL0ONvv4fWJYEU6Gv4qzY'

   auth=OAuthHandler(consumer_key,consumer_secret)
   auth.set_access_token(access_token,access_secret)
   args=['facebook']


   list_tweets=[]
   query=request.form['keyword']
   api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
   # collect tweets on     #facebook
   for tweet in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang="en").items(100):
       #print (tweet.text.encode('utf-8'))
       list_tweets.append(tweet.text)
   #print(list_tweets)
       

   with open('model.pickle','rb') as f:
       vectorizer=pickle.load(f)

   with open('classifier.pickle','rb') as f:
       clf=pickle.load(f)

   total_pos=0
   total_neg=0

   process_tweets=[]

   #preprocessing the tweets
   for tweet in list_tweets:
       tweet=re.sub(r"^https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
       tweet=re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
       tweet=re.sub(r"\s+https://t.co/a-zA-Z0-9]*$"," ",tweet)
       tweet=tweet.lower()
       tweet=re.sub(r"that's","that is",tweet)
       tweet=re.sub(r"there's","there is",tweet)
       tweet=re.sub(r"what's","what is",tweet)
       tweet=re.sub(r"where's","where is",tweet)
       tweet=re.sub(r"it's","it is",tweet)
       tweet=re.sub(r"who's","who is",tweet)
       tweet=re.sub(r"i'm","i am",tweet)
       tweet=re.sub(r"she's","she is",tweet)
       tweet=re.sub(r"he's","he is",tweet)
       tweet=re.sub(r"they're","they are",tweet)
       tweet=re.sub(r"who're","who are",tweet)
       tweet=re.sub(r"ain't","am not",tweet)
       tweet=re.sub(r"wouldn't","wound not",tweet)
       tweet=re.sub(r"can't","can not",tweet)
       tweet=re.sub(r"could't","could not",tweet)
       tweet=re.sub(r"won't","will not",tweet)
       tweet=re.sub(r"should't","should not",tweet)
       tweet=re.sub(r"\W"," ",tweet)
       tweet=re.sub(r"\d"," ",tweet)
       tweet=re.sub(r"\s+[a-z]\s"," ",tweet)
       tweet=re.sub(r"\s+[a-z]$"," ",tweet)
       tweet=re.sub(r"^[a-z]\s"," ",tweet)
       tweet=re.sub(r"\s+"," ",tweet)
       sent=clf.predict(vectorizer.transform([tweet]).toarray())
       process_tweets.append(tweet)
       #print(tweet,":",sent)
       if sent[0]==1:
           total_pos+=1
       else:
           total_neg+=1

   objects=['Positive','Negative']
   y_pos=np.arange(len(objects))

   plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
   plt.xticks(y_pos,objects)
   plt.ylabel('Number')
   plt.title('Number of Positive and Negative Tweets')
   #plt.show()
   #print(clf.predict(vectorizer.transform(['Srilakshmi and Jhansi are friends'])))                         
   return render_template("get_tweets.html",query=query,list_tweets=list_tweets,process_tweets=process_tweets)

if __name__ == '__main__':
   app.run(debug=True)
