import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.random.set_seed(0)
np.random.seed(0)


save_dir = './figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def load_data():
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.load('E:\\mnist-experient\\mnist\\train.npy')
    y_train = np.load('E:\\mnist-experient\\mnist\\train-label.npy')
    # reshape train data to vector
    x_train = x_train.reshape(x_train.shape[0], -1)
    # normalize to [-1, 1]
    x_train = x_train/127.5 - 1.0
    # convert to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    return x_train, y_train


def build_generator():
    """
    建立生成器
    """
    generator = tf.keras.Sequential()
    generator.add(layers.Dense(256, input_shape=(110,)))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dense(512))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dense(1024))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dense(784, activation='tanh'))
    return generator


def build_discriminator_base():
    """
    定义判别器basenet
    """
    discriminator_base = tf.keras.Sequential()
    discriminator_base.add(layers.Dense(512, input_shape=(784,)))
    discriminator_base.add(layers.BatchNormalization())
    discriminator_base.add(layers.ReLU())
    discriminator_base.add(layers.Dense(256))
    discriminator_base.add(layers.BatchNormalization())
    discriminator_base.add(layers.ReLU())
    return discriminator_base


class Generator(models.Model):

    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.generator = build_generator()

    def call(self, xs, training=True):
        return self.generator(xs)


class Discriminator(models.Model):

    def __init__(self, num_classes=10, **kwargs):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.discriminator_base = build_discriminator_base()
        # 有监督前景分类
        self.classifer1 = layers.Dense(self.num_classes, activation='softmax')
        # 真假分布分类
        self.logit = layers.Dense(1, activation='sigmoid')
        # 包含背景分类
        self.classifer2 = layers.Dense(self.num_classes+1, activation='softmax')

    def call(self, xs, training=True):
        xs = self.discriminator_base(xs)
        cls1_output = self.classifer1(xs)
        distrib_output = self.logit(xs)
        cls2_output = self.classifer2(xs)
        return cls1_output, cls2_output, distrib_output


class LabelGAN(models.Model):
    """
    定义完整的LabelGAN网络用于训练生成器
    """
    def __init__(self, **kwargs):
        super(LabelGAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        # freeze discriminator
        self.discriminator.trainable = False

    def call(self, xs, training=True):
        xs = self.generator(xs, training)
        cls_output, distrib_output = self.discriminator(xs, training)
        return cls_output, distrib_output


def show_figure(figures, epoch):
    # reshape
    figures = figures.reshape(figures.shape[0], 28, 28)
    figures = (figures+1.0)/2
    plt.figure(figsize=(10, 10))
    for i in range(figures.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(figures[i], cmap='gray')
        plt.axis('off')
    plt.savefig('./figures/epoch-%d.png'%epoch)
    plt.close()

@ tf.function
def train_generator_one_step(noise, real_labs, optimizer, generator, discriminator):
    # alpha = np.minimum(0.1+(50-epoch)/50, 1.0)
    with tf.GradientTape() as tape:
        noise_input = tf.concat([noise, real_labs], axis=-1)
        g_labs = tf.ones((noise.shape[0], 1))
        fake_imgs = generator(noise_input)
        cls_outputs, _, distrib_output = discriminator(fake_imgs)
        cls_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(real_labs, cls_outputs))
        distrib_loss = tf.reduce_mean(tf.losses.binary_crossentropy(g_labs, distrib_output))
        g_loss = cls_loss + distrib_loss
        # compute gradients
        grads = tape.gradient(g_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return g_loss


@ tf.function
def train_discriminator_one_step(noise, real_imgs, real_labs, optimizer, generator, discriminator):
    with tf.GradientTape() as tape:
        noise_input = tf.concat([noise, real_labs], axis=-1)
        fake_imgs = generator(noise_input)
        d_inputs = tf.concat([real_imgs, fake_imgs], axis=0)
        # 包含背景类分类器label
        b_fake_labs = tf.zeros_like(real_labs)
        b_real_labs = tf.concat([real_labs, tf.zeros((real_labs.shape[0], 1))], axis=1)
        b_fake_labs = tf.concat([b_fake_labs, tf.ones((b_fake_labs.shape[0], 1))], axis=1)
        b_labs = tf.concat([b_real_labs, b_fake_labs], axis=0)
        # 判别器用于判别真假分布的标签
        d_labs = np.zeros((noise.shape[0]*2,1), dtype=np.float32)
        d_labs[:noise.shape[0]] = 1.0
        d_labs = tf.convert_to_tensor(d_labs)
        cls1_outputs, cls2_outputs, distrib_output = discriminator(d_inputs)
        # 计算一个bs样本loss
        cls1_loss = 2.0*tf.reduce_mean(tf.losses.categorical_crossentropy(real_labs, cls1_outputs[:noise.shape[0]]))
        # 计算二个bs样本loss
        distrib_loss = tf.reduce_mean(tf.losses.binary_crossentropy(d_labs, distrib_output))
        cls2_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(b_labs, cls2_outputs))
        d_loss = cls1_loss + distrib_loss + cls2_loss
        # compute gradients
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    return d_loss


def train():
    epochs=100
    batchsize=128
    # build discriminator
    discriminator = Discriminator()
    generator = Generator()
    # build optimizer
    optimizer_g = tf.optimizers.Adam(2e-4, beta_1=0.5)
    optimizer_d = tf.optimizers.Adam(2e-4, beta_1=0.5)
    # load data
    x_train, y_train = load_data()

    for epoch in range(epochs):
        for step in range(x_train.shape[0]//batchsize):
            noise = tf.random.uniform(shape=(batchsize, 100), minval=0.0, maxval=1.0)
            idx = np.random.choice(x_train.shape[0], batchsize, replace=False)
            real_imgs = tf.gather(x_train, idx)
            real_labs = tf.gather(y_train, idx)
            # train discriminator
            d_loss = train_discriminator_one_step(noise, real_imgs, real_labs, optimizer_d, generator, discriminator)

            # train generator
            noise = tf.random.uniform(shape=(batchsize, 100), minval=0.0, maxval=1.0)
            g_loss = train_generator_one_step(noise, real_labs, optimizer_g, generator, discriminator)

            if step%10 == 0:
                print('Epoch:[{:03d}]-Generator:{:.4f}-Discriminator:{:.4f}'.format(epoch+1, g_loss, d_loss))

        # predict
        noise = np.random.uniform(size=(100, 100)).astype(np.float32)
        labs = np.array(list(range(10))*10)
        labs = tf.keras.utils.to_categorical(labs, 10)
        noise_input = np.hstack((noise, labs))
        g_imgs = generator(noise_input, training=False)
        show_figure(g_imgs.numpy(), epoch+1)


if __name__ == "__main__":
    train()