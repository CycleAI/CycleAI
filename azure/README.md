# Azure
Code for training models on Azure.


## Running 

To run the code add the pre processed images to a folder, (in this example we will use `image_pixels`) and the txt file with the safety ratings of the images (`sorted_imgs.txt`) for this case.

The file `inference.py` requires as arguments first the folder with the images and next the txt file, example:

```
python3 inference.py ./images_pixels/ ./sorted_imgs.txt 
```

The models will be trained and sent to the [WANDB dashboard](https://wandb.ai/gonvas_/cycleai?workspace=user-warcraft12321) along with the weights. 
