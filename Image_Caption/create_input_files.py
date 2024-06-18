from utils import create_input_files_custom

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files_custom(dataset='coco',
                              karpathy_json_path='data/coco/dataset_coco.json',
                              image_folder='data/train2014/',
                              captions_per_image=5,
                              min_word_freq=5,
                              output_folder='data/coco/',
                              max_len=50)
