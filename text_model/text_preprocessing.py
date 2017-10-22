import os
import re
#import gensim

import numpy as np
import pandas as pd

# string.punctuation
_PUNCTUATION = u'!"$%&\'()*+,./:;<=>?[\\]^_`{|}~#'

_MIN_ENGLISH_WORDS_IN_POST = 5

def _load_embedding_weights_glove(text_dir, emb_dir, filename):
    """Load the word embedding weights from a pre-trained model.
    
    Parameters:
        text_dir: The directory containing the text model.
        emb_dir: The subdirectory containing the weights.
        filename: The name of that text file.
        
    Returns:
        vocabulary: A list containing the words in the vocabulary.
        embedding: A numpy array of the weights.
    """
    vocabulary = []
    embedding = []
    with open(os.path.join(text_dir, emb_dir, filename), 'rb') as f:
        for line in f.readlines():
            row = line.strip().split(' ')
            # Convert to unicode
            vocabulary.append(row[0].decode('utf-8', 'ignore'))
            embedding.append(map(np.float32, row[1:]))
        embedding = np.array(embedding)
        print('Finished loading word embedding weights.')
    return vocabulary, embedding

def _load_embedding_weights_word2vec(text_dir, emb_dir, filename):
    """Load the word embedding weights from a pre-trained model.
    
    Parameters:
        text_dir: The directory containing the text model.
        emb_dir: The subdirectory containing the weights.
        filename: The name of the binary file.
        
    Returns:
        vocabulary: A list containing the words in the vocabulary.
        embedding: A numpy array of the weights.
    """
    word2vec_dir = os.path.join(text_dir, emb_dir, filename)
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_dir, binary=True)
    vocabulary = model.index2word
    embedding = model.syn0
    print('Finished loading word embedding weights.')
    return vocabulary, embedding

def _str_list_to_set(str_list):
    """Convert a string representation of a list such as '[happy, sun, outdoors]'
       to a set of strings {'happy', 'sun', 'outdoors'}
    """
    output = str_list[1:-1].split(',')
    output = set([x.strip() for x in output])
    return output

def _df_with_hashtag_in_post(df, tag):
    """Make sure that the relevant hashtag is in the post.
    """
    df['tags'] = df['tags'].map(_str_list_to_set)
    mask = df['tags'].map(lambda x: tag in x)
    return df.loc[mask, :].reset_index(drop=True)

def _is_valid_text(paragraph, vocab_set):
    """Check that a post contains atleast _MIN_ENGLISH_WORDS_IN_POST words in english.
    """
    # Check for nan text
    if (type(paragraph) == float) and (np.isnan(paragraph)):
        return False
    else:
        regex = re.compile('[%s]' % re.escape(_PUNCTUATION))
        # Remove punctuation, convert to lower case before splitting
        words = regex.sub('', paragraph).lower().split()
        # Check if there are atleast _MIN_ENGLISH_WORDS_IN_POST words in english
        return len(set(words).intersection(vocab_set)) > _MIN_ENGLISH_WORDS_IN_POST

def _paragraph_to_ids(paragraph, word_to_id, post_size, emotions):
    """Convert a paragraph to a list of ids, removing the #emotion.
    """
    words = []
    vocab_size = len(word_to_id)

    # Remove emotion hashtags from the post.
    emotion_regex = re.compile('|'.join(map(re.escape, ['#' + emotion for emotion in emotions])))
    paragraph = emotion_regex.sub('', paragraph.lower())

    regex = re.compile('[%s]' % re.escape(_PUNCTUATION))
    # Remove punctuation, convert to lower case before splitting
    words = regex.sub('', paragraph).lower().split()
    # Replace unknown words by an id equal to the size of the vocab
    words = map(lambda x: word_to_id.get(x, vocab_size), words)
    words_len = len(words)
    if words_len > post_size:
        words = words[:post_size]
        words_len = post_size
    else:
        words = words + [vocab_size] * (post_size - words_len)
    return words, words_len

def preprocess_df(text_dir, emb_dir, filename, emb_name, emotions, post_size):
    """Preprocess emotion dataframes.
    """
    if emb_name == 'word2vec':
        vocabulary, embedding = _load_embedding_weights_word2vec(text_dir, emb_dir, filename)
    else:
        vocabulary, embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)
    vocab_size, embedding_dim = embedding.shape
    word_to_id = dict(zip(vocabulary, range(vocab_size)))
    # Unknown words = vector with zeros
    embedding = np.concatenate([embedding, np.zeros((1, embedding_dim))])

    columns = ['id', 'post_url', 'type', 'timestamp', 'date', 'tags', 'liked',
               'note_count', 'photo', 'text', 'search_query']
    df_all = pd.DataFrame(columns=columns)
    for emotion in emotions:
        path = os.path.join('data', emotion + '.csv')
        df_emotion = _df_with_hashtag_in_post(pd.read_csv(path, encoding='utf-8'), emotion)
        df_all = pd.concat([df_all, df_emotion]).reset_index(drop=True)

    vocab_set = set(vocabulary)
    mask = df_all['text'].map(lambda x: _is_valid_text(x, vocab_set))
    df_all =  df_all.loc[mask, :].reset_index(drop=True)

    # Map text to ids
    df_all['text_list'], df_all['text_len'] = zip(*df_all['text'].map(lambda x: 
        _paragraph_to_ids(x, word_to_id, post_size, emotions)))

    # Binarise emotions
    emotion_dict = dict(zip(emotions, range(len(emotions))))
    df_all['search_query'] =  df_all['search_query'].map(emotion_dict)

    # Add <ukn> word to dictionary
    word_to_id['<ukn>'] = vocab_size
    print('Finished loading dataframes.')

    return df_all, word_to_id, embedding

def preprocess_one_df(vocabulary, embedding, emotion, post_size):
    """Preprocess one dataframe for the image/text model.
    """
    vocab_size, embedding_dim = embedding.shape
    word_to_id = dict(zip(vocabulary, range(vocab_size)))
    # Unknown words = vector with zeros
    #embedding = np.concatenate([embedding, np.zeros((1, embedding_dim))])

    path = os.path.join('data', emotion + '.csv')
    df_emotion = _df_with_hashtag_in_post(pd.read_csv(path, encoding='utf-8'), emotion)

    vocab_set = set(vocabulary)
    mask = df_emotion['text'].map(lambda x: _is_valid_text(x, vocab_set))
    df_emotion =  df_emotion.loc[mask, :].reset_index(drop=True)

    emotions = ['happy', 'sad', 'angry', 'scared', 'disgusted', 'surprised']
    # Map text to ids
    df_emotion['text_list'], df_emotion['text_len'] = zip(*df_emotion['text'].map(lambda x: 
        _paragraph_to_ids(x, word_to_id, post_size, emotions)))

    # Binarise emotions
    #emotion_dict = dict(zip(emotions, range(len(emotions))))
    #df_all['search_query'] =  df_all['search_query'].map(emotion_dict)

    # Add <ukn> word to dictionary
    #word_to_id['<ukn>'] = vocab_size

    return df_emotion#, word_to_id, embedding

