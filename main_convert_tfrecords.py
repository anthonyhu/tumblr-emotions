import sys
from datasets.convert_images_tfrecords import convert_images_with_text

if __name__ == '__main__':
    args = sys.argv[1:]
    num_valid = (int)(args.pop(0))

    if len(args) > 0:
        sys.stderr.write('Too many arguments given.\n')
    else:
        dataset_dir = 'data'
        convert_images_with_text(dataset_dir, num_valid)