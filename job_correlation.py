# Import necessary packages
from image_text_model.im_text_rnn_model import correlation_matrix

nb_batches = 781
checkpoint_dir = 'image_text_model/deep_sentiment_model'
posts_logits, posts_labels = correlation_matrix(nb_batches, checkpoint_dir)

# Save output and parameters to text file in the localhost node, which is where the computation is performed.
#with open('/data/localhost/not-backed-up/ahu/jobname_' + str(slurm_id) + '_' + str(slurm_parameter) + '.txt', 'w') as text_file:
	#text_file.write('Parameters: {0} Result: {1}\n'.format(parameter, output))