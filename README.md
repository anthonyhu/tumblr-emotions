# Multimodal Sentiment Analysis to Explore the Structure of Emotions (KDD 2018)
This repository contains the code to reproduce the results in the paper: [link].

## Tumblr data
In the `datasets` module:
* `tumblr_extraction.py` extracts Tumblr posts (url, date, image url, text, tags).
* `download_images.py` downloads the images using the image url.
* `convert_images_tfrecords.py` converts the posts to the TFRecord format for the TensorFlow Dataset pipeline.

## Training the model
* The Image model can be trained with `train_image_model` in the `image_model` module.
* The Text model can be trained with `train_text_model` in the `text_model` module.
* The Deep Sentiment model can be trained with `train_deep_sentiment` in the `image_text_model` module.

## Results 
* The hierarchical clustering can be obtained using `correlation_matrix` in the `image_text_model` module.
* The top words can be obtained using `word_most_relevant` in the `image_text_model` module.
* The OASIS correlation can be obtained using `oasis_evaluation` in the `image_text_model` module.


