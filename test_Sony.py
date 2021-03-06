# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from PIL import Image
import time
import pickle

tf.compat.v1.disable_v2_behavior()
tf.keras.backend.clear_session()

input_dir = './dataset/Sony/test/'
gt_dir = './dataset/Sony/gt/'
checkpoint_dir = './checkpoint/'
result_dir = './final_results/'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

# PSNR, SSIM and taken_time arrays
PSNR_array = []
SSIM_array = []
taken_time = []

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = [int('10003')]


def storeData(psnr, ssim, taken_time):
    with open('psnr.pkl', 'wb') as f:
        pickle.dump(psnr, f)

    with open('ssim.pkl', 'wb') as f:
        pickle.dump(ssim, f)

    with open('taken_time.pkl', 'wb') as f:
        pickle.dump(taken_time, f)


def loadData():
    with open('psnr.pkl', 'rb') as f:
        psnr = pickle.load(f)

    with open('ssim.pkl', 'rb') as f:
        ssim = pickle.load(f)

    with open('taken_time.pkl', 'rb') as f:
        taken_time = pickle.load(f)

    return psnr, ssim, taken_time


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal(
        [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(
        x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(
        pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(
        conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(
        pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(
        conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(
        pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(
        conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(
        conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(
        conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(
        conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    print('Network is defined\n\n\n')
    return out


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    print('raw data packed\n\n\n')
    return out


def input_to_png(raw, test_id, ratio):
    im = raw.raw_image_visible.astype(np.float32)
    # subtract the black level
    im = np.maximum(im - 512, 0) / (16383 - 512)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    im = np.concatenate((im[0:H:2, 0:W:2, :],
                        im[0:H:2, 1:W:2, :],
                        im[1:H:2, 1:W:2, :]), axis=2)

    im = np.expand_dims(np.float32(im / 65535.0), axis=0)
    im = im[0, :, :, :]
    Image.fromarray((im*255).astype('uint16'), mode='RGB').save(result_dir +
                                                                'final/%5d_00_%d_input.png' % (test_id, ratio))


#PSNR_array, SSIM_array, taken_time = loadData()
print('Loaded pickle files with sizes {} for PSNR, {} for SSIM, and {} for time_taken'.format(
    len(PSNR_array), len(SSIM_array), len(taken_time)))


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for test_id in test_ids[20:]:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % (test_id))
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
        input_full = np.minimum(input_full, 1.0)

        output = sess.run(out_image, feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output, 0), 1)
        output = output[0, :, :, :]

        taken_time.append(time.time() - st)

        input_to_png(raw, test_id, ratio)

        im = raw.postprocess(use_camera_wb=True, half_size=False,
                             no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(
            use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
        gt_full = gt_full[0, :, :, :]

        scale_full = scale_full[0, :, :, :]
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the groundtruth

        # Compute PSNR & SSIM for each image
        im1 = tf.image.convert_image_dtype(output, tf.float32)
        im2 = tf.image.convert_image_dtype(gt_full, tf.float32)
        psnr = tf.image.psnr(im1, im2, max_val=1.0)
        ssim = tf.image.ssim(im1, im2, max_val=1.0)
        with tf.Session() as ses:
            PSNR_array.append(ses.run(psnr))

        with tf.Session() as ses:
            SSIM_array.append(ses.run(ssim))

        print("Current PSNR is: %.3f and SSIM is: %.3f with time of: %.3f" %
              (PSNR_array[-1], SSIM_array[-1], taken_time[-1]))

        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
        scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
        scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))

storeData(PSNR_array, SSIM_array, taken_time)
print("PSNR is: %.3f and SSIM is: %.3f with average time of: %.3f" %
      (np.mean(PSNR_array), np.mean(SSIM_array), np.mean(taken_time)))
