# Blind Removal of Facial Foreign Shadows
Official Github repository for "Blind Removal of Facial Foreign Shadows", accepted to BMVC 2022. 

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
DATA_DIR_TEST = ['sample_imgs/*']
```
to 
```
DATA_DIR_TEST = ['UCB/*']
```
and 
```
fsr.testFFHQ(dataset_test)
```
to
```
fsr.test(dataset_test)
```

## SFW Dataset
Link to the original SFW video dataset: https://drive.google.com/file/d/1H4WFtDCGp4Bk1EeTjX1F_LXDmxxlxUHm/view?usp=share_link
