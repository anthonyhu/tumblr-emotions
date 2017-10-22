import os
import operator

import numpy as np
import pandas as pd
from text_model.text_preprocessing import _load_embedding_weights_glove, preprocess_one_df
from image_text_model.im_text_rnn_model import word_most_relevant
from datasets.dataset_utils import read_label_file

emotions = ['happy', 'sad', 'scared', 'angry', 'surprised', 'disgusted', 'annoyed', 'bored', 
'love', 'calm', 'amazed', 'optimistic', 'pensive', 'ashamed', 'excited'] #removed interested

df_dict = dict()

text_dir = 'text_model'
emb_dir = 'embedding_weights'
filename = 'glove.6B.50d.txt'
vocabulary, embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)
_POST_SIZE = 50
    
for emotion in emotions:
    df = preprocess_one_df(vocabulary, embedding, emotion, _POST_SIZE)
    df_dict[emotion] = df
    
columns = [u'id', u'post_url', u'type', u'timestamp',
           u'date', u'tags', u'liked', u'note_count',
           u'photo', u'text', u'search_query', u'text_list',
           u'text_len']
df_all = pd.DataFrame(columns = [])

for emotion in emotions:
    df_all = pd.concat([df_all, df_dict[emotion]]).reset_index(drop=True)

# Get top 1000 most occurring words
vocabulary_count = dict(zip(vocabulary, np.zeros(len(vocabulary), dtype=np.int32)))
for i in range(df_all.shape[0]):
    for word in df_all['text'][i].lower().split():
        if word in vocabulary_count:
            vocabulary_count[word] += 1

nb_top_words = 1000
most_frequent_words = sorted(vocabulary_count.items(), key=operator.itemgetter(1), reverse=True)[:nb_top_words]

word_to_id = dict(zip(vocabulary, range(len(vocabulary))))
id_most_frequent_words = [word_to_id[x[0]] for x in most_frequent_words]

labels_dict = read_label_file('data', 'photos')
num_classes = len(labels_dict)
checkpoint_dir = 'image_text_model/deep_sentiment_model'
scores, vocabulary, word_to_id = word_most_relevant(id_most_frequent_words, num_classes, checkpoint_dir)