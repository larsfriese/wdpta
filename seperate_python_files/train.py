from numpy import array
import tensorflow, datetime, pandas
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

max_length = 60
vocab_size = 80
lookup_input = {}

df = pandas.read_csv('data.csv')
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
print(y)

# Model:

model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x, y, epochs=100, verbose=0)

# Evaluation:

loss, accuracy = model.evaluate(x, y, verbose=0)
print('Accuracy: %f /100' % (accuracy*100))
model.save('ml_model_sen', '/')