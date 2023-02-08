# Blind Removal of Facial Foreign Shadows
Official Github repository for "Blind Removal of Facial Foreign Shadows", accepted to BMVC 2022. 

[Yaojie Liu*](https://yaojieliu.github.io/), [Andrew Hou*](https://andrewhou1.github.io/), [Xinyu Huang](https://scholar.google.com/citations?user=cL4bNBwAAAAJ&hl=en), [Liu Ren](https://sites.google.com/site/liurenshomepage/), [Xiaoming Liu](http://www.cse.msu.edu/~liuxm/index2.html) (* denotes equal contribution). 

![alt text](https://github.com/andrewhou1/BlindShadowRemoval/BMVC_2022_teaser.png)

## Dependencies
Our method has the following dependencies:

1. Tensorflow 2.3.0
2. Python 3.7.10
3. OpenCV 4.7.0
4. NumPy 1.18.5
5. Tensorflow keras 2.4.0
6. sklearn 0.24.2
7. scipy 1.4.1
8. matplotlib 3.4.2 
9. PIL 8.2.0
10. skimage 0.18.1
11. tensorflow_addons 0.13.0
12. os, glob, time, sys, random, math, natsort, face-alignment

## Preprocessing New Images
Before feeding an image to our method, use **bmvc2022_dataprocess.py** to compute landmarks for all images in the **sample_uncropped_images/** folder. You can empty this folder initially and just place all of your images in (you only need .png, it will generate the .npy files that contain the landmarks). 

Next, run **dataprocess.py**. This will generate a folder for each image in the **sample_uncropped_images_cropped/** folder, where each folder contains the landmarks and a cropped image. 

## Testing Procedures 
To test on **in-the-wild images**, place the resulting folder with your cropped image and landmarks in the **sample_imgs/** folder and run:
```
python train_test_GSC.py
```

To run the **SFW evaluation** on shadow segmentation performance (AUC) and shadow removal performance, run:
```
python train_with_TSM.py
```

To test on the **UCB test set**, also run: 
```
python train_test_GSC.py
```
You will need to change two lines in this file first. Change 
```
DATA_DIR_TEST = ['sample_imgs/*'] to DATA_DIR_TEST = ['UCB/train/input/*']
```
and 
```
fsr.testFFHQ(dataset_test) to fsr.test(dataset_test)
```
You will also need to change one line in **dataset.py**
```
Change dataset = dataset.map(map_func=self.parse_fn_test_FFHQ, num_parallel_calls=autotune) to dataset = dataset.map(map_func=self.parse_fn_test, num_parallel_calls=autotune)

```

## SFW Dataset
Link to the original SFW video dataset: https://drive.google.com/file/d/1H4WFtDCGp4Bk1EeTjX1F_LXDmxxlxUHm/view?usp=share_link

## Citation 
If you utilize our code in your work, please cite our BMVC 2022 paper. 
```
@inproceedings{ blind-removal-of-facial-foreign-shadows,
  author = { Yaojie Liu* and Andrew Hou* and Xinyu Huang and Liu Ren and Xiaoming Liu },
  title = { Blind Removal of Facial Foreign Shadows },
  booktitle = { In Proceedings of British Machine Vision Conference (BMVC) },
  address = { London, UK },
  month = { November },
  year = { 2022 },
}
```

## Contact 
If there are any questions, please feel free to post here or contact the first author at **houandr1@msu.edu** 
