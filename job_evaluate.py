# Import necessary packages
import os
#from image_model.im_model import evaluate_model_2
from image_text_model.im_text_rnn_model import evaluate_deep_sentiment

# Obtain the environment variables to determine the parameters
slurm_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
slurm_parameter = int(os.environ['SLURM_ARRAY_TASK_ID'])

#checkpoint_dir = 'image_model/fine_tuned_model'
#log_dir = 'image_model/model_eval'
checkpoint_dir = 'image_text_model/deep_sentiment_model_trunc'
log_dir = 'image_text_model/model_eval_trunc'
num_evals = 50
if slurm_parameter == 0:
	mode = 'train'
else:
	mode = 'validation'
evaluate_deep_sentiment(checkpoint_dir, log_dir, mode, num_evals)

# Save output and parameters to text file in the localhost node, which is where the computation is performed.
#with open('/data/localhost/not-backed-up/ahu/jobname_' + str(slurm_id) + '_' + str(slurm_parameter) + '.txt', 'w') as text_file:
	#text_file.write('Parameters: {0} Result: {1}\n'.format(parameter, output))