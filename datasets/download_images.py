import urllib
import os
import pandas as pd
import tensorflow as tf

def download_im(search_query, start, end, dataset_dir, subdir='photos'):
    """Download images given the urls in the dataframe specified by a search query.

    Args:
        search_query: A string giving the sentiment of the photos to be downloaded.
        start: Index to start downloading.
        end: Index to finish downloading.
        dataset_dir: Directory where the dataframes are stored.
        subdir: Subdirectory to store the photos.

    Returns:
        Images downloaded in the directory dataset_dir/subdir/search_query, having 
        the posts ids as names.
    """
    # Load previously scraped data
    df = pd.read_csv(os.path.join(dataset_dir, search_query + '.csv'), encoding='utf-8')
    links = df['photo']
    ids = df['id']
    photos_dir = os.path.join(dataset_dir, subdir, search_query)
    if not tf.gfile.Exists(photos_dir):
        tf.gfile.MakeDirs(photos_dir)
    for i in range(start, end):
        # Check for NaNs
        if links[i] == links[i]:
            # Download photo
            urllib.urlretrieve(links[i], os.path.join(photos_dir, str(ids[i]) + '.jpg'))