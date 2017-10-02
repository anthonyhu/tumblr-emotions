# Import necessary packages
import os
import numpy as np
from datasets.download_images import download_im_with_text

# Obtain the environment variables to determine the parameters
slurm_id = int(os.environ['SLURM_ARRAY_JOB_ID'])
slurm_parameter = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Our parameters we will like to grid search over
emotions = ['happy', 'sad', 'scared', 'angry', 'surprised', 'disgusted']
df_sizes = {'happy': 116930,
            'sad': 66747, 
            'angry': 59899, 
            'surprised': 8625,
            'scared': 67756,
            'disgusted': 6349}

search_query = 'happy'#emotions[slurm_parameter]
indices = np.linspace(9000, 60000, 7, dtype=int)
start = indices[slurm_parameter]
end = indices[slurm_parameter + 1]
download_im_with_text(search_query, start, end, 'data')

# Save output and parameters to text file in the localhost node, which is where the computation is performed.
#with open('/data/localhost/not-backed-up/ahu/jobname_' + str(slurm_id) + '_' + str(slurm_parameter) + '.txt', 'w') as text_file:
	#text_file.write('Parameters: {0} Result: {1}\n'.format(parameter, output))