# Instruction

Train for AI Upscaling
```
# train the SRCNN model using GPU, set learning rate=0.0005, batch size=256,
# make the program train 100 epoches and save a checkpoint every 10 epoches
:\> python train.py train --cuda --lr=0.0005 --batch-size=256 --num-epoch=100 --savefreq=10

# train the SRCNN model using CPU, set learning rate=0.001, batch size=128,
# make the program train 20 epoches and save a checkpoint every 2 epoches
:\> python train.py train --lr=0.001 --batch-size=128 --num-epoch=20 --save-freq=2

# resume training with GPU from "checkpoint.x" with saved hyperparameters
:\> python train.py resume checkpoint.x --cuda
# resume training from "checkpoint.x" and override some of saved hyperparameters
:\> python train.py resume checkpoint.x --batch-size=16 --num-epoch=200

# inspect "checkpoint.x"
:\> python train.py inspect checkpoint.x 
```

Run the code
```
:\> python main.py
or
:\> python main.py --cuda
```
# Option Menu
![image](https://user-images.githubusercontent.com/39010822/165690601-ed1dedc6-fc3d-4b90-8379-650951e24121.png)

2nd and 3rd option run HDR tone mapping on HDR images, then upscaled the toned image.

HDR image: **test image/hdr_images**

HDR Panorama folder: test **image/hdr_panorama/inputs/**

# Panorama Page

3rd option menu, select **checkpoint.best** as trained model and **test image/hdr_panorama/inputs/** for HDR images input

Move the slider to change the scale of Image Upscale, Default is 3x

![image](https://user-images.githubusercontent.com/39010822/165690080-41217153-46d0-4db8-a476-2bbcf0678583.png)
