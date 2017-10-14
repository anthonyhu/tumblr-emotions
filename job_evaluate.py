# Import necessary packages
import os
from image_model.im_model import evaluate_image_model
from text_model.text_embedding import evaluate_text_model
from image_text_model.im_text_rnn_model import evaluate_deep_sentiment

# Obtain the environment variables to determine the parameters
slurm_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
slurm_parameter = int(os.environ['SLURM_ARRAY_TASK_ID'])

num_evals = 50

if slurm_parameter == 0:
	mode = 'train'
else:
	mode = 'validation'

model_type = 'image'

if model_type == 'image':
	checkpoint_dir = 'image_model/finetuned_image_model'
	log_dir = 'image_model/image_model_eval'
	evaluate_image_model(checkpoint_dir, log_dir, mode, num_evals)
elif model_type == 'text':
	checkpoint_dir = 'text_model/trained_text_model'
	log_dir = 'text_model/text_model_eval'
	evaluate_text_model(checkpoint_dir, log_dir, mode, num_evals)
elif model_type == 'image_text':
	checkpoint_dir = 'image_text_model/deep_sentiment_model'
	log_dir = 'image_text_model/model_eval'
	evaluate_deep_sentiment(checkpoint_dir, log_dir, mode, num_evals)

# Save output and parameters to text file in the localhost node, which is where the computation is performed.
#with open('/data/localhost/not-backed-up/ahu/jobname_' + str(slurm_id) + '_' + str(slurm_parameter) + '.txt', 'w') as text_file:
	#text_file.write('Parameters: {0} Result: {1}\n'.format(parameter, output))