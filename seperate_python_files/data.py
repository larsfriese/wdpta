import twint, pandas
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

print(len(final))
print(final)