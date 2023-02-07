# Blind Removal of Facial Foreign Shadows
Official Github repository for "Blind Removal of Facial Foreign Shadows", accepted to BMVC 2022. 

## Dependencies
Our method has the following dependencies:

1. Tensorflow 2.3.0
2. Python 3.7.10
3. OpenCV 3.4.2
4. NumPy 1.18.5
5. Tensorflow keras 2.4.0
6. sklearn 0.24.2
7. scipy 1.4.1
8. matplotlib 3.4.2 
9. PIL 8.2.0
10. skimage 0.18.1
11. tensorflow_addons 0.13.0
12. os, glob, time, sys, random, math, natsort

## Preprocessing a New Image
Before feeding an image to our method, use 

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
