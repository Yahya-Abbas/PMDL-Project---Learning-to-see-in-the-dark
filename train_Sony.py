# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import rawpy
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image

import os
import time
import glob


input_dir = './dataset/Sony/train/'
gt_dir = './dataset/Sony/gt/'
checkpoint_dir = './result_Sony/'
result_dir = './result_Sony/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
validation_fns = glob.glob(gt_dir + '2*.ARW')
validation_ids = [int(os.path.basename(validation_fn)[0:5])
                  for validation_fn in validation_fns]
ps = 512  # patch size for training
save_freq = 50

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


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

    # apply white balance, normalize white balance coefficients to the 2nd coefficient, which is ususally the coefficient for green
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
""" def second_encoder(input):
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

    return conv5 """


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
def decoder(first_encoder_output):
    up6 = upsample_and_concat(
        first_encoder_output[4], first_encoder_output[3], 256, 512)
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


def network(tradPipeline_input_image):

    first_encoder_output = first_encoder(tradPipeline_input_image)

    #second_encoder_output = second_encoder(tradPipeline_input_image)

    """ concatenated_latent_space = tf.concat(
        [first_encoder_output[4], second_encoder_output], -1) """
    #concatenated_latent_space.set_shape([None, None, None, output_channels * 2])

    # Here I give the decoder only the first encoder output, should be changed to the concatenation of
    # Both encoder outputs
    decoder_output = decoder(first_encoder_output)

    #print('model defined ')
    # print('')
    # print('')
    return decoder_output


training_loss = []
validation_loss = []


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

    return out


sess = tf.Session()

#in_image = tf.placeholder(tf.float32, [None, None, None, 4])
tradPipeline_image = tf.placeholder(tf.float32, [None, None, None, 3])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

out_image = network(tradPipeline_image)

#G_loss_PSNR = tf.reduce_mean(20*log10(gt_image)-10*log10(tf.abs((out_image - gt_image))))
G_loss_MSE = tf.reduce_mean(out_image - gt_image)
G_loss_MAE = tf.reduce_mean(tf.abs(out_image - gt_image))
G_loss = G_loss_MAE

t_vars = tf.compat.v1.trainable_variables()
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#print (checkpoint_dir)

if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * 60

#tradPipeline_images = [None] * 60

tradPipeline_images = {}
tradPipeline_images['300'] = [None] * 60
tradPipeline_images['250'] = [None] * 60
tradPipeline_images['100'] = [None] * 60

# Losses
g_loss = np.zeros((120, 1))
val_loss = np.zeros((15, 1))

# print(result_dir)
#allfolders = glob.glob(result_dir + '*0')
lastepoch = 0


# for folder in allfolders:
#    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
for epoch in range(lastepoch, 4001):
    print(epoch)
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0  # counter of the images processes in this epoch

    if epoch > 1200:  # after 30 epochs decrease the learning rate to move slower and avoid overshooting the minimum
        learning_rate = 1e-5

    losses = []
    # randomly pick images from the training set
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        #print (in_files)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()

        cnt += 1

        if (ind >= len(tradPipeline_images[str(ratio)[0:3]])) or (tradPipeline_images[str(ratio)[0:3]][ind] is None):
            raw = rawpy.imread(in_path)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

            if ind < len(tradPipeline_images[str(ratio)[0:3]]):
                tradPipeline_images[str(ratio)[0:3]][ind] = np.expand_dims(
                    trad_pipeline(raw, ratio=ratio), axis=0)

                #tradPipeline_images[ind] = np.expand_dims(trad_pipeline(raw, ratio=ratio), axis=0)

                gt_images[ind] = np.expand_dims(
                    np.float32(im / 65535.0), axis=0)
            else:
                #desk_input_image = np.expand_dims(pack_raw(raw), axis=0) * ratio

                desk_tradPipeline_image = np.expand_dims(
                    trad_pipeline(raw, ratio=ratio), axis=0)

                desk_gt_image = np.expand_dims(
                    np.float32(im / 65535.0), axis=0)

            #print('rawpy processed')

        # crop

        H = tradPipeline_images[str(ratio)[0:3]][ind].shape[1] if ind < len(
            tradPipeline_images[str(ratio)[0:3]]) else desk_tradPipeline_image.shape[1]
        W = tradPipeline_images[str(ratio)[0:3]][ind].shape[2] if ind < len(
            tradPipeline_images[str(ratio)[0:3]]) else desk_tradPipeline_image.shape[2]
        #print('images cropped')

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        tradPipeline_patch = tradPipeline_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :] if ind < len(
            tradPipeline_images[str(ratio)[0:3]]) else desk_tradPipeline_image[:, yy:yy + ps, xx:xx + ps, :]

        """ tradPipeline_patch = tradPipeline_images[ind][:, yy:yy + ps, xx:xx + ps, :] if ind < len(
            input_images[str(ratio)[0:3]]) else desk_tradPipeline_image[:, yy:yy + ps, xx:xx + ps, :] """

        #print('input images defined')
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :] if ind < len(
            tradPipeline_images[str(ratio)[0:3]]) else desk_gt_image[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]
        #print('gt images defined')

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            #input_patch = np.flip(input_patch, axis=1)
            tradPipeline_patch = np.flip(tradPipeline_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            #input_patch = np.flip(input_patch, axis=2)
            tradPipeline_patch = np.flip(tradPipeline_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            #input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            tradPipeline_patch = np.transpose(tradPipeline_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        tradPipeline_patch = np.minimum(tradPipeline_patch, 1.0)

        #print('session starts')
        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={tradPipeline_image: tradPipeline_patch, gt_image: gt_patch, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)

        #print('output calculated')
        g_loss[ind] = G_current
        print("Epoch: %d, Training Loss = %.3f, Time=%.3f" %
              (epoch, np.mean(g_loss[np.where(g_loss)]), time.time() - st))
        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)
            temp = np.concatenate(
                (gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))
            #Image.fromarray((temp*255).astype('uint8'), mode='L').convert('RGB').save(result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

    for ind in np.random.permutation(len(validation_ids)):
        # get the path from image id

        val_id = validation_ids[ind]
        in_filesV = glob.glob(input_dir + '%05d_00*.ARW' % val_id)
        #print (in_files)
        in_pathV = in_filesV[np.random.random_integers(0, len(in_filesV) - 1)]
        in_fn = os.path.basename(in_pathV)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % val_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        raw = rawpy.imread(in_pathV)

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(
            use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

        #val_input_image = np.expand_dims(pack_raw(raw), axis=0) * ratio

        val_tradPipeline_image = np.expand_dims(
            trad_pipeline(raw, ratio=ratio), axis=0)

        val_gt_image = np.expand_dims(
            np.float32(im / 65535.0), axis=0)

        #print('rawpy processed')

        #print('session starts')
        val_output_image, curr_val_loss = sess.run([out_image, G_loss],
                                                   feed_dict={tradPipeline_image: val_tradPipeline_image, gt_image: val_gt_image})

        val_loss[ind] = (curr_val_loss)

        val_output_image = np.minimum(np.maximum(val_output_image, 0), 1)

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)
            temp = np.concatenate(
                (val_gt_image[0, :, :, :], val_output_image[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%05d_00_validation_%d.jpg' % (epoch, val_id, ratio))

    # Compute epoch loss
    training_loss.append(np.mean(g_loss[np.where(g_loss)]))
    validation_loss.append(np.mean(val_loss[np.where(val_loss)]))
    print("Epoch: %d, Training Loss = %.3f, Validation Loss = %0.3f, Time=%.3f" % (
        epoch, training_loss[-1], validation_loss[-1], time.time() - st))
    # training_loss.append(np.array(losses).mean())
    # training_loss.append(losses)

    saver.save(sess, checkpoint_dir + 'model.ckpt')

    if epoch % save_freq == 0:
        plt.figure()
        plt.plot(range(1, len(training_loss)+1), training_loss, 'r--')
        plt.plot(range(1, len(validation_loss)+1),
                 validation_loss, 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('./plots/epoch%04d_losses.png' % (epoch+1))

plt.figure()
plt.plot(range(1, len(range(lastepoch, 4001))+1), training_loss, 'r--')
plt.plot(range(1, len(range(lastepoch, 4001))+1), validation_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
