Sample landscapes folder:


```
landscapes   
│
└───city
│   │   14407411024_b5dfc2d8e2.jpg
│   │   315774020_55984b813f.jpg
│   │   ...
│   field
│   │   ...
│   forest   
│   │   ...
│   lake
│   │   ...
│   mountain
│   │   ...
│   ocean
│   │   ...
│   road
└───  

```

To train on the emotion landscapes dataset:
```
python train_gan.py --image_width <image_width>
```
Image width can be 32 or 64.

Multi-scale conditional vae training:
``` 
python train_mscvae.py --beta 0.00005 --all_labels --z_dim 8 --image_width 64 --nlevels 4
```
