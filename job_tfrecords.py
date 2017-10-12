# Import necessary packages
from datasets.convert_images_tfrecords import convert_images_with_text

dataset_dir = 'data'
num_valid = 50000
convert_images_with_text(dataset_dir, num_valid)

# Save output and parameters to text file in the localhost node, which is where the computation is performed.
#with open('/data/localhost/not-backed-up/ahu/jobname_' + str(slurm_id) + '_' + str(slurm_parameter) + '.txt', 'w') as text_file:
	#text_file.write('Parameters: {0} Result: {1}\n'.format(parameter, output))