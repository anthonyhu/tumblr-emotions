import os
import pandas as pd
import tensorflow as tf
import urllib2
import io
from PIL import Image

def download_im(search_query, start, end, dataset_dir, subdir='photos'):
    """Download images given the urls in the dataframe specified by the search query.

    Parameters:
        search_query: A string giving the sentiment to load the corresponding dataframe.
        start: A start index for the loaded dataframe.
        end: An end index for the loaded dataframe.
        dataset_dir: A directory where the dataframes are stored.
        subdir: A subdirectory to store the photos.

    Returns:
        Images downloaded in the directory dataset_dir/subdir/search_query, having 
        the posts ids as names.
    """
    # Load data
    df = pd.read_csv(os.path.join(dataset_dir, search_query + '.csv'), encoding='utf-8')
    links = df['photo']
    ids = df['id']
    # Create subdir if it doesn't exist
    if not tf.gfile.Exists(os.path.join(dataset_dir, subdir)):
        tf.gfile.MakeDirs(os.path.join(dataset_dir, subdir))
    # Create search_query folder if it doesn't exist
    photos_dir = os.path.join(dataset_dir, subdir, search_query)
    if not tf.gfile.Exists(photos_dir):
        tf.gfile.MakeDirs(photos_dir)
    for i in range(start, end):
        # Check for NaNs
        if links[i] == links[i]:
            # Open url and convert to JPEG image
            try:
                f = urllib2.urlopen(links[i])
            except Exception:
                continue
            image_file = io.BytesIO(f.read())
            im = Image.open(image_file)
            filename = str(ids[i]) + '.jpg'
            im.convert('RGB').save(os.path.join(photos_dir, filename), 'JPEG')