from __future__ import division
import numpy as np
import rawpy
from PIL import Image
import os
import time
import glob

gt_dir = './dataset/Sony/gt/'
train_dir = './dataset/Sony/train/'

png_gt = './dataset/jpg_images/gt/'
png_train = './dataset/jpg_images/train/'

gt_images_fns = glob.glob(gt_dir + '*.ARW')
gt_images_ids = [int(os.path.basename(train_fn)[0:5])
                 for train_fn in gt_images_fns]

train_images_fns = glob.glob(train_dir + '*.ARW')
train_images_ids = [int(os.path.basename(train_fn)[0:5])
                    for train_fn in train_images_fns]

# Convert gt photos to png
for ind in range(len(gt_images_ids)):
    image_id = gt_images_ids[ind]
    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % image_id)
    gt_path = gt_files[0]
    gt_fn = os.path.basename(gt_path)

    gt_exposure = float(gt_fn[9:-5])

    gt_raw = rawpy.imread(gt_path)
    im = gt_raw.postprocess(
        use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=16)

    im = np.float32(im / 65535.0)
    #im = np.expand_dims(np.float32(im / 65535.0), axis=0)
    #im = im[0, :, :, :]
    Image.fromarray((im*255).astype('uint8'), mode='RGB').save(png_gt +
                                                               '%05d_00_%ds.jpg' % (image_id, gt_exposure))


# Convert train photos to png
for ind in range(len(train_images_ids)):
    image_id = train_images_ids[ind]
    train_files = glob.glob(train_dir + '%05d_00*.ARW' % image_id)
    for train_path in train_files:
        train_fn = os.path.basename(train_path)

        train_exposure = train_fn[9:-4]

        train_raw = rawpy.imread(train_path)
        im = train_raw.raw_image_visible.astype(np.float32)
        # subtract the black level
        im = np.maximum(im - 512, 0) / (16383 - 512)
        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        im = np.concatenate((im[0:H:2, 0:W:2, :],
                            im[0:H:2, 1:W:2, :],
                            im[1:H:2, 1:W:2, :]), axis=2)
        Image.fromarray((im*255).astype('uint8'), mode='RGB').save(png_train +
                                                                   '%05d_00_%s.jpg' % (image_id, train_exposure))
