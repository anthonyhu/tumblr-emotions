import sys

from time import time
from image_model.im_model import forest

if __name__ == '__main__':
	args = sys.argv[1:]
	num_valid = (int)(args.pop(0))
	n_estimators = (int)(args.pop(0))
	max_depth = (int)(args.pop(0))

	if len(args) > 0:
		std.stderr.write('Too many arguments given.\n')
	else:
		t0 = time()
		forest(num_valid, n_estimators, max_depth)
		print('The training took : {0:.1f} mins'.format((time() - t0) / 60))