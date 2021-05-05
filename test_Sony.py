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


# The traditional pipeline algorithm we adopted
def trad_pipeline(raw_image, ratio=100):
    image = np.array(raw_image.raw_image_visible, dtype=np.double)

    # subtract black levels and normalize to interval [0..1]
    black = np.reshape(
        np.array(raw_image.black_level_per_channel, dtype=np.double), (2, 2))
    black = np.tile(black, (image.shape[0]//2, image.shape[1]//2))
    image = np.maximum(image - 512, 0) / (16383 - 512)

    # find the positions of the three (red, green and blue) or four base colors within the Bayer pattern
    n_colors = raw.num_colors
    colors = np.frombuffer(raw_image.color_desc, dtype=np.byte)
    pattern = np.array(raw_image.raw_pattern)
    index_0 = np.where(colors[pattern] == colors[0])
    index_1 = np.where(colors[pattern] == colors[1])
    index_2 = np.where(colors[pattern] == colors[2])
    index_3 = np.where(colors[pattern] == colors[3])

    # apply white balance, normalize white balance coefficients to the 2nd coefficient, which is usually the coefficient for green
    wb_c = raw_image.camera_whitebalance
    wb = np.zeros((2, 2), dtype=np.double)
    wb[index_0] = wb_c[0] / wb_c[1]
    wb[index_1] = wb_c[1] / wb_c[1]
    wb[index_2] = wb_c[2] / wb_c[1]
    if n_colors == 4:
        wb[index_3] = wb_c[3] / wb_c[3]
    wb = np.tile(wb, (image.shape[0]//2, image.shape[1]//2))
    image_wb = np.clip(image * wb * ratio, 0, 1)

    # demosaic
    image_demosaiced = np.empty(
        (image_wb.shape[0]//2, image_wb.shape[1]//2, n_colors))
    if n_colors == 3:
        im = np.expand_dims(image_wb, axis=2)
        img_shape = image_wb.shape
        H = img_shape[0]
        W = img_shape[1]
        image_demosaiced = np.concatenate((im[0:H:2, 0:W:2, :],
                                           im[0:H:2, 1:W:2, :],
                                           im[1:H:2, 1:W:2, :]), axis=2)
    else:  # n_colors == 4
        im = np.expand_dims(image_wb, axis=2)
        img_shape = image_wb.shape
        H = img_shape[0]
        W = img_shape[1]
        image_demosaiced = np.concatenate((im[0:H:2, 0:W:2, :],
                                           im[0:H:2, 1:W:2, :],
                                           im[1:H:2, 1:W:2, :]), axis=2)

    # convert to linear sRGB, calculate the matrix that transforms sRGB into the camera's primary color components and invert this matrix to perform the inverse transformation
    XYZ_to_cam = np.array(raw.rgb_xyz_matrix[0:n_colors, :], dtype=np.double)
    sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], dtype=np.double)
    sRGB_to_cam = np.dot(XYZ_to_cam, sRGB_to_XYZ)
    norm = np.tile(np.sum(sRGB_to_cam, 1), (3, 1)).transpose()
    sRGB_to_cam = sRGB_to_cam / norm
    if n_colors == 3:
        cam_to_sRGB = np.linalg.inv(sRGB_to_cam)
    else:  # n_colors == 4
        cam_to_sRGB = np.linalg.pinv(sRGB_to_cam)
    # performs the matrix-vector product for each pixel
    image_sRGB = np.einsum('ij,...j', cam_to_sRGB, image_demosaiced)
    # apply sRGB gamma curve
    i = image_sRGB < 0.0031308
    j = np.logical_not(i)
    image_sRGB[i] = 323 / 25 * image_sRGB[i]
    image_sRGB[j] = 211 / 200 * image_sRGB[j] ** (5 / 12) - 11 / 200
    image_sRGB = np.clip(image_sRGB, 0, 1)

    # scipy.misc.toimage(image_sRGB * 255, high=255, low=0, cmin=0, cmax=255).save('/home/ml2/Desktop/traditional pipeline outputs/test.jpg')

    return image_sRGB


# define the first encoder that takes the raw image with simple preprocessing
def first_encoder(input):
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

    return conv1, conv2, conv3, conv4, conv5

# define the encoder after the traditional algorithm pipeline


def second_encoder(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g2_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g2_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g2_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g2_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(
        pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g2_conv3_1')
    conv3 = slim.conv2d(
        conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g2_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(
        pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g2_conv4_1')
    conv4 = slim.conv2d(
        conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g2_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(
        pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g2_conv5_1')
    conv5 = slim.conv2d(
        conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g2_conv5_2')

    return conv5


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal(
        [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(
        x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


# We will need to change the arguments given to the upsample_and_concat function
def decoder(latent_space, first_encoder_output):
    up6 = upsample_and_concat(latent_space, first_encoder_output[3], 256, 1024)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(
        conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, first_encoder_output[2], 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(
        conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, first_encoder_output[1], 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, first_encoder_output[0], 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(
        conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)

    return out


def network(raw_input_image, tradPipeline_input_image):

    first_encoder_output = first_encoder(raw_input_image)

    second_encoder_output = second_encoder(tradPipeline_input_image)

    concatenated_latent_space = tf.concat(
        [first_encoder_output[4], second_encoder_output], -1)
    #concatenated_latent_space.set_shape([None, None, None, output_channels * 2])

    # Here I give the decoder only the first encoder output, should be changed to the concatenation of
    # Both encoder outputs
    decoder_output = decoder(concatenated_latent_space, first_encoder_output)

    #print('model defined ')
    # print('')
    # print('')
    return decoder_output


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

    Image.fromarray((im*255).astype('uint8'), mode='RGB').save(result_dir +
                                                               'final/%5d_00_%d_input.png' % (test_id, ratio))


# PSNR_array, SSIM_array, taken_time = loadData()
# print('Loaded pickle files with sizes {} for PSNR, {} for SSIM, and {} for time_taken'.format(len(PSNR_array), len(SSIM_array), len(taken_time)))


sess = tf.Session()

in_image = tf.placeholder(tf.float32, [None, None, None, 4])
tradPipeline_image = tf.placeholder(tf.float32, [None, None, None, 3])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

out_image = network(in_image, tradPipeline_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for test_id in test_ids[48:]:
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

        raw = rawpy.imread(in_path)

        st = time.time()

        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
        input_full = np.minimum(input_full, 1.0)

        with tf.Session() as ses:
            trad_pipeline_input = np.expand_dims(
                trad_pipeline(raw, ratio=ratio), axis=0)

        output = sess.run(out_image, feed_dict={
                          in_image: input_full, tradPipeline_image: trad_pipeline_input})

        taken_time.append(time.time() - st)

        output = np.minimum(np.maximum(output, 0), 1)
        output = output[0, :, :, :]

        input_to_png(raw, test_id, ratio)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(
            use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
        gt_full = gt_full[0, :, :, :]

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

        scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))

storeData(PSNR_array, SSIM_array, taken_time)
print("PSNR is: %.3f and SSIM is: %.3f with average time of: %.3f" %
      (np.mean(PSNR_array), np.mean(SSIM_array), np.mean(taken_time)))
