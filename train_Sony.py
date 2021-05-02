# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os
import time
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import cv2

input_dir = './dataset/jpg_images/train/'
gt_dir = './dataset/jpg_images/gt/'
checkpoint_dir = './results_jpg/'
result_dir = './results_jpg/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.jpg')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
validation_fns = glob.glob(gt_dir + '2*.jpg')
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
    return out


training_loss = []
validation_loss = []


sess = tf.Session()

in_image = tf.placeholder(tf.float32, [None, None, None, 3])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

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

if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Keep images in memory after loading to save time.
gt_images = [None] * 135
input_images = {}
input_images['300'] = [None] * 60
input_images['250'] = [None] * 60
input_images['100'] = [None] * 60

# Losses
g_loss = np.zeros((1000, 1))
val_loss = np.zeros((1000, 1))


allfolders = glob.glob(result_dir + '*0')
lastepoch = 0

for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
for epoch in range(lastepoch, 4001):
    print(epoch)
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0  # counter of the images processes in this epoch

    if epoch > 500:  # after 30 epochs decrease the learning rate to move slower and avoid overshooting the minimum
        learning_rate = 1e-5

    losses = []
    # randomly pick images from the training set
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id

        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.jpg' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.jpg' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if (ind >= len(input_images[str(ratio)[0:3]])) or (input_images[str(ratio)[0:3]][ind] is None):

            im = cv2.imread(in_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            gt_im = cv2.imread(gt_path)
            gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)

            if ind < len(input_images[str(ratio)[0:3]]):
                input_images[str(ratio)[0:3]][ind] = np.expand_dims(
                    im, axis=0) * ratio

                gt_images[ind] = np.expand_dims(
                    np.float32(gt_im/255.0), axis=0)
            else:
                desk_input_image = np.expand_dims(im, axis=0) * ratio

                desk_gt_image = np.expand_dims(np.float32(gt_im/255.0), axis=0)

            #print('rawpy processed')

        # crop

        H = input_images[str(ratio)[0:3]][ind].shape[1] if ind < len(
            input_images[str(ratio)[0:3]]) else desk_input_image.shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2] if ind < len(
            input_images[str(ratio)[0:3]]) else desk_input_image.shape[2]
        #print('images cropped')

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :] if ind < len(
            input_images[str(ratio)[0:3]]) else desk_input_image[:, yy:yy + ps, xx:xx + ps, :]

        #print('input images defined')
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :] if ind < len(
            input_images[str(ratio)[0:3]]) else desk_gt_image[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]
        #print('gt images defined')

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        #print('session starts')
        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)

        # input_images[str(ratio)[0:3]][ind] = None
        # gt_images[ind] = None

        #print('output calculated')
        g_loss[ind] = G_current

        print("Epoch: %d, Training Loss = %.3f, Time=%.3f" %
              (epoch, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)
            temp = np.concatenate(
                (gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            #scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))
            Image.fromarray((temp*255).astype('uint8'), mode='RGB').save(
                result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))
            #Image.fromarray((temp*255).astype('uint8'), mode='L').convert('RGB').save(result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

    for ind in np.random.permutation(len(validation_ids)):
        # get the path from image id

        val_id = validation_ids[ind]
        in_filesV = glob.glob(input_dir + '%05d_00*.jpg' % val_id)
        in_pathV = in_filesV[np.random.random_integers(0, len(in_filesV) - 1)]
        in_fn = os.path.basename(in_pathV)

        gt_files = glob.glob(gt_dir + '%05d_00*.jpg' % val_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        val_im = cv2.imread(in_pathV)
        val_im = cv2.cvtColor(val_im, cv2.COLOR_BGR2RGB)

        gt_im = cv2.imread(gt_path)
        gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)

        val_input_image = np.expand_dims(val_im, axis=0) * ratio

        val_gt_image = np.expand_dims(np.float32(gt_im/255.0), axis=0)

        #print('session starts')
        val_output_image, curr_val_loss = sess.run([out_image, G_loss],
                                                   feed_dict={in_image: val_input_image, gt_image: val_gt_image})

        #print('Current validation loss is {}'.format(curr_val_loss))
        val_loss[ind] = curr_val_loss
        #print('Validation loss array is {}'.format(val_loss))
        val_output_image = np.minimum(np.maximum(val_output_image, 0), 1)

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)
            temp = np.concatenate(
                (val_gt_image[0, :, :, :], val_output_image[0, :, :, :]), axis=1)
            #scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_validation_%d.jpg' % (epoch, val_id, ratio))
            Image.fromarray((temp*255).astype('uint8'), mode='RGB').save(
                result_dir + '%04d/%05d_00_validation_%d.jpg' % (epoch, val_id, ratio))

    # Compute epoch loss and graph the plot if save epoch
    training_loss.append(np.mean(g_loss[np.where(g_loss)]))
    validation_loss.append(np.mean(val_loss[np.where(val_loss)]))
    print("Epoch: %d, Training Loss = %.3f, Validation Loss = %0.3f, Time=%.3f" %
          (epoch, training_loss[-1], validation_loss[-1], time.time() - st))

    if epoch % save_freq == 0:
        plt.figure()
        plt.plot(range(1, len(training_loss)+1), training_loss, 'r--')
        plt.plot(range(1, len(validation_loss)+1),
                 validation_loss, 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.xticks(range(1, len(training_loss)+1))
        plt.savefig('./plots/epoch%04d_losses.png' % (epoch+1))

    saver.save(sess, checkpoint_dir + 'model.ckpt')

plt.figure()
plt.plot(range(1, len(range(lastepoch, 4001))+1), training_loss, 'r--')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, len(range(lastepoch, 4001))+1))
plt.show()
