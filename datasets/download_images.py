import urllib
import os
import pandas as pd
import tensorflow as tf

def download_im(search_query, start, end):
    """Download images given the urls in the dataframe specified by the search_query

    Args:
        search_query: A string giving the sentiment to load the corresponding dataframe
        start: index to start downloading
        end: index to end downloading

    Returns:
        Images downloaded in the directory data/photos/search_query, having the post id
        as names.
    """
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