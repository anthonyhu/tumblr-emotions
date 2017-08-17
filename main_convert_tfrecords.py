from datasets.convert_images_tfrecords import convert_images

if __name__ == '__main__':
	dataset_dir = 'data'
	convert_images(dataset_dir)