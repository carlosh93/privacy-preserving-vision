# Image caption
## 1. Dataset
### Coco dataset

1. Download the dataset [coco](https://cocodataset.org/#download) 
2. Create **./data/coco** path in **Image_caption/** and put the dataset folder(s)
3. Run **create_input_files.py** 
4. Download [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and put **dataset_coco.json** file in **./data/coco** path 

### Flickr8k dataset

1. Download the dataset [flickr8k](https://www.kaggle.com/adityajn105/flickr8k/activity)
2. Create **./data/flickr8k** path in **Image_caption/** and put the dataset folder(s)
3. Run **create_input_files.py**
4. Download [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and put **dataset_flickr8k.json** file in **./data/flickr8k** path 


### Output example
```
...
- train.py
- data/
  - coco
    - train2014/
    - dataset_coco.json
    - TEST_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json
    - TEST_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json
    - TEST_IMAGES_coco_5_cap_per_img_5_min_word_freq.json
    - VAL_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json
    - VAL_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json
    - VAL_IMAGES_coco_5_cap_per_img_5_min_word_freq.json
    - TRAIN_CAPLENS_coco_5_cap_per_img_5_min_word_freq.json
    - TRAIN_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json
    - TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.json
    - WORDMAP_coco_5_cap_per_img_5_min_word_freq
		
		
```

## 2. Train
We conduct training for a total of 20 epochs with a batch size of 64. To fine-tune the learning process, we utilize the following learning rates: encoder learning rate at 1e-4, decoder learning rate at 5e-4, and camera learning rate at 5e-7. We use *pytorch 1.8.0* and *torchvision 0.9.0*.
You will **require** a camera heating [Model.pth](https://drive.google.com/drive/folders/1Ex5AiuCfQXa_LnxDwGlw9t8zRp3_hDhF?usp=sharing) to achieve better results, then put the file in **./Camera** path. 

Run *train.py*

### Pretrained files
To run with the pre-trained weights download our [Checkpoint](https://drive.google.com/drive/folders/1Ex5AiuCfQXa_LnxDwGlw9t8zRp3_hDhF?usp=sharing) and put them into **./results/GPU_final** path. 

## 3. Test
We validate our method trought different metrics including BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, and Cider. This code facilitates the extraction of 1000 privacy-preserving images from a dataset and the retrieval of their corresponding captions, while also providing metrics such as PSNR, MSE, and the model's performance evaluations.
```
cd eval 
python eval_total.py
```
To obtain a privacy protected image while obtaining its associated caption, you can input an image in the following manner:

```
cd eval
python caption.py --img path_to_your_image.jpg
```