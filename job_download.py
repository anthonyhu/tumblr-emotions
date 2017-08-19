# Import necessary packages
import os
from datasets.download_images import download_im

# Obtain the environment variables to determine the parameters
slurm_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
slurm_parameter = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Our parameters we will like to grid search over
emotions = ['happy', 'sad', 'scared', 'angry', 'surprised', 'disgusted']

search_query = emotions[slurm_parameter]
download_im(search_query, 0, 9000, 'data')

# Save output and parameters to text file in the localhost node, which is where the computation is performed.
#with open('/data/localhost/not-backed-up/ahu/jobname_' + str(slurm_id) + '_' + str(slurm_parameter) + '.txt', 'w') as text_file:
	#text_file.write('Parameters: {0} Result: {1}\n'.format(parameter, output))