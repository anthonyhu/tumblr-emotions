import urllib
import os
import pandas as pd
import tensorflow as tf

def download_im(search_query, start, end):
    # Load previously scraped data
    df = pd.read_csv(os.path.join('data', search_query + '.csv'), encoding='utf-8')
    links = df['photo']
    ids = df['id']
    dataset_dir = os.path.join('data/photos', search_query)

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    for i in range(start, end):
        # Check for NaN
        if links[i] == links[i]:
            urllib.urlretrieve(links[i], os.path.join(dataset_dir, str(ids[i]) + '.jpg'))