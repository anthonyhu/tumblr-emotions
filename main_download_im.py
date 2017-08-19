import sys
from datasets.download_images import download_im

if __name__ == '__main__':
    args = sys.argv[1:]
    search_query = args.pop(0)
    start = (int)(args.pop(0))
    end = (int)(args.pop(0))

    if len(args) > 0:
        sys.stderr.write('Too many arguments given.\n')
    else:
        dataset_dir = 'data'
        # This might issue a warning about EXIL data corrupted
        # but these info are not useful for our task anyway.
        download_im(search_query, start, end, dataset_dir)

