from image_text_model.im_text_rnn_model import outliers_detection

# change mode to validation and set nb_batches
checkpoint_dir = 'image_text_model/deep_sentiment_model'
max_norms, max_post_ids = outliers_detection(checkpoint_dir)