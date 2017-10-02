# Import necessary packages
import os
import tensorflow as tf
from image_model.im_model import download_pretrained_model
#from image_model.im_model import fine_tune_model_with_text
from image_text_model.im_text_model import train_deep_sentiment

# Obtain the environment variables to determine the parameters
slurm_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
slurm_parameter = int(os.environ['SLURM_ARRAY_TASK_ID'])

num_steps = 10
initial_lr = 0.001

# Maybe download pre-trained model
checkpoints_dir = 'image_model/pretrained_model'
if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)
    url = 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'
    download_pretrained_model(url, checkpoints_dir)

# Fine-tune model
dataset_dir = 'data'
#train_dir = 'image_model/fine_tuned_model'
train_dir = 'image_text_model/deep_sentiment_model'
train_deep_sentiment(dataset_dir, checkpoints_dir, train_dir, num_steps=num_steps, 
                     initial_lr=initial_lr)

# Save output and parameters to text file in the localhost node, which is where the computation is performed.
#with open('/data/localhost/not-backed-up/ahu/jobname_' + str(slurm_id) + '_' + str(slurm_parameter) + '.txt', 'w') as text_file:
	#text_file.write('Parameters: {0} Result: {1}\n'.format(parameter, output))