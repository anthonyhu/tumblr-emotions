import urllib
import os
import pandas as pd
import tensorflow as tf
from PIL import Image

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
    # Load data
    df = pd.read_csv(os.path.join(dataset_dir, search_query + '.csv'), encoding='utf-8')
    links = df['photo']
    ids = df['id']
    photos_dir = os.path.join(dataset_dir, subdir, search_query)
    if not tf.gfile.Exists(photos_dir):
        tf.gfile.MakeDirs(photos_dir)
    for i in range(start, end):
        # Check for NaNs
        if links[i] == links[i]:
            # The filename is: id + image extension
            extension = links[i].split('.')[-1]
            filename = str(ids[i]) + '.' + extension
            # Download photo
            urllib.urlretrieve(links[i], os.path.join(photos_dir, filename))
            # Convert to .jpg if necessary and delete old filename
            if extension != 'jpg':
                im = Image.open(os.path.join(photos_dir, filename))
                new_filename = str(ids[i]) + '.jpg'
                im.convert('RGB').save(os.path.join(photos_dir, new_filename), 'JPEG')
                os.remove(os.path.join(photos_dir, filename))