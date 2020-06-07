from numpy import array
import tensorflow, datetime
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import twint
from collections import Counter

from train import encode_matrix

# How are people in [location] feeling about [keyword] this week?
keyword = 'trump'
location='all'
final=[]
tweets=[]

def get_tweets(keyword, location):
	d = datetime.timedelta(days = 7)
	dt = datetime.datetime.today() - d
	c = twint.Config()
	c.Search = keyword
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
print(Counter(opinions))