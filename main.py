from numpy import array
import tensorflow, datetime, pandas, twint
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, Dropout
from collections import Counter

max_length = 60
vocab_size = 400
lookup_input = {}

def make_data():
    popular_users=['realDonaldTrump','BarackObama','rihanna','nytimes','elonmusk']
    final=[]

    def get_tweets_from_user(user, keyword, limit):
        tweets=[]
        c = twint.Config()
        if keyword != '': c.Search = keyword
        c.Since = '2020-03-01 00:00:00'
        c.Lang = 'en'
        c.Username = user
        c.Store_object = True
        c.Store_object_tweets_list = tweets
        c.Limit = limit
        twint.run.Search(c)
        return tweets

    for i in popular_users:
        l = get_tweets_from_user(i, '', 10)
        for x in l: final.append(x.tweet)

    for c,i in enumerate(final, 0): # truncate tweets to a certain length
        t=' '.join(i.split()[:60])
        final[c]=t

    df = pandas.DataFrame(final, columns=['tweet'])
    df['sad'] = 0
    df['angry'] = 0
    df['scared'] = 0
    df['happy'] = 0
    df['surprised'] = 0
    df['disgusted'] = 0
    df.to_csv('data.csv')

def train(data_csv):

    df = pandas.read_csv(data_csv)
    docs_x = df['tweet'].tolist()
    docs_y=[]
    for index, row in df.iterrows():
        temp = [row['sad'], row['angry'], row['scared'], row['happy'], row['surprised'], row['disgusted']]
        docs_y.append(temp)

    # define documents
    #y = array([[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0]])

    #sad, andgry, scared/feared, happy, surprised, disgusted
    #[0, 0, 0, 0, 0, 0]
    
    def encode_matrix(list_of_strings):
        encoded_docs = [one_hot(d, vocab_size) for d in list_of_strings]
        for i,x in zip(encoded_docs, docs_x):
            lookup_input[x]=i
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        return padded_docs

    x = encode_matrix(docs_x)
    y = array(docs_y)

    # Model:
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=100, verbose=0)

    # Evaluation:
    loss, accuracy = model.evaluate(x, y, verbose=0)
    print('Accuracy: %f /100' % (accuracy*100))
    model.save('ml_model_sen', '/')

def predict(keyword, location):
    # How are people in [location] feeling about [keyword] this week?
    final=[]
    tweets=[]

    def get_tweets(keyword, location):
        d = datetime.timedelta(days = 7)
        dt = datetime.datetime.today() - d
        c = twint.Config()
        c.Search = keyword
        c.Near = location
        c.Since = str(dt.strftime("%Y-%m-%d %H:%M:%S"))
        c.Lang = 'en'
        c.Limit = 100
        c.Store_object = True
        c.Store_object_tweets_list = tweets
        twint.run.Search(c)
        return tweets

    l = get_tweets(keyword, location)
    for x in l: final.append(x.tweet) # twint format to standart list with strings
    for c,i in enumerate(final, 0): # truncate tweets to a certain length
        t=' '.join(i.split()[:60])
        final[c]=t

    #for i in final: print(f'\n\n{len(i.split())}\n\n') check length of tweets
    #temp=[]
    #temp2=0
    #for i in final: temp.append(len(i))
    #for i in temp: temp2+=i
    #print(temp2/len(temp)) #check average words per tweets
    
    def encode_matrix(list_of_strings):
        encoded_docs = [one_hot(d, vocab_size) for d in list_of_strings]
        for i,x in zip(encoded_docs, final):
            lookup_input[x]=i
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        return padded_docs

    opinions=[]
    model = tensorflow.keras.models.load_model('ml_model_sen')
    results = model.predict(encode_matrix(final)).tolist()
    for i in results:
        n=i.index(max(i))
        if n==0:
            opinions.append('sad')
        elif n==1:
            opinions.append('angry')
        elif n==2:
            opinions.append('scared')
        elif n==3:
            opinions.append('happy')
        elif n==4:
            opinions.append('surprised')
        elif n==5:
            opinions.append('disgusted')
    print(f'\n\nHow are people in {location} feeling about {keyword} this week?')
    print(Counter(opinions))

if __name__ == '__main__': 
    train('data.csv')
    predict('floyd','newyork')