import tensorflow as tf

from DataReader import *

sample_rate = 16000
frame_length = 255
frame_step = 124
log_offset = 1e-9

n_classes = 12

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha*x)

def Conv2dLayer(number, input_data, filters, kernel_size, reuse=False):
    with tf.variable_scope("conv2d_layer_{}".format(number), reuse=reuse):
        conv = tf.layers.conv2d(input_data, filters=filters, kernel_size=kernel_size, padding='same')
        lrl = leaky_relu(conv)
        pool = tf.layers.max_pooling2d(lrl, 2, 2, padding='same')
        return pool

class Model(object):

    def __init__(self):

        self.train = tf.placeholder(tf.bool, shape=[])
        self.input = tf.placeholder(tf.float32, [None, 3, 16000])
        self.label = tf.placeholder(tf.int64, [None])
        self.cls_lr = tf.placeholder(tf.float32, shape=[])
        split_each = tf.split(self.input, axis=1, num_or_size_splits=3)

        self.feature_vector = self.Conv2dModel(self.LogMagSpecCom(split_each[0]))
        for i in range(1, 3):
            self.feature_vector += self.Conv2dModel(self.LogMagSpecCom(split_each[i]), reuse=True)

        self.cls_vector = tf.layers.dense(self.feature_vector, 12)

        self.classify_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label, logits=self.cls_vector
        ))
        self.cls_optimizer = tf.train.AdamOptimizer(self.cls_lr).minimize(self.classify_loss)

        self.predict = tf.argmax(self.cls_vector, axis=1)

        self.saver = tf.train.Saver()
        self.sess = None

    def Model_Init(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)

    def Model_Close(self):
        self.sess.close()

    def LogMagSpecCom(self, input):
        wave_signal = tf.squeeze(input, axis=1)
        stfts = tf.contrib.signal.stft(wave_signal, frame_length=frame_length,
                                            frame_step=frame_step, fft_length=frame_length)
        magnitude_spectrograms = tf.abs(stfts)
        log_magnitude_spectrogram = tf.log(magnitude_spectrograms + log_offset)
        # bs 127 128
        return log_magnitude_spectrogram

    def Conv2dModel(self, input, reuse=False):
        with tf.variable_scope('conv2d_model', reuse=reuse):
            input_layer = tf.reshape(input, shape=(-1, 127, 128, 1))
            conv1 = Conv2dLayer(1, input_layer, 8, 7, reuse)
            # bs 64 64 8
            conv2 = Conv2dLayer(2, conv1, 16, 5, reuse)
            # bs 32 32 16
            conv3 = Conv2dLayer(3, conv2, 32, 5, reuse)
            # bs 16 16 32
            conv4 = Conv2dLayer(4, conv3, 64, 3, reuse)
            # bs 8 8 64
            conv5 = Conv2dLayer(5, conv4, 128, 3, reuse)
            # bs 4 4 128
            conv6 = Conv2dLayer(6, conv5, 256, 3, reuse)
            # bs 2 2 256
            return tf.layers.batch_normalization(tf.layers.flatten(conv6), training=self.train)

