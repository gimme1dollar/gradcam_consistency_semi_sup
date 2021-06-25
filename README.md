# Semi-supervised Image Classification with Grad-CAM Consistency

## Basic information about framework
- backbone network : efficientnet-b04 
- augmentation : crop & color jitter
- consistency : prediction + grad-CAM


## Preparation 
> python3 main_baseline.py 

- You may follow instructions in https://github.com/kinux98/DLProjectGroup12


## Validation & kaggle submission (DL20 Dataset)
> python3 test.py --pretrained-ckpt="./checkpoints/1_50_FINAL/best.pth" --exp-name="1_50_FINAL"

## Contributors
이주용 gimme1dollar       
권동현 kinux98
