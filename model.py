from keras.models import Input, Model
from keras.layers import Add, Dense, Conv2D, Concatenate, MaxPooling2D, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization

'''
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
CBAM: Convolutional Block Attention Module
(https://arxiv.org/pdf/1807.06521.pdf)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
res_rate: rate at which the residual blocks repeat
ratio: ratio of MLP with one hidden layer
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each attention block if true
dilated: change convolution to dilate convolution for fast train if true
'''


def get_unetcbam_model(input_channel_num=3, out_ch=3, start_ch=16, depth=5, inc_rate=2., activation='relu',
         dropout=0.5, ratio=4, batchnorm=True, maxpool=False, upconv=True, residual=False, dilated=False):
    def _conv_block(m, dim, acti, bn, di, do=0):
        n = Conv2D(dim, 3, padding='same', dilation_rate=2)(m) if di else Conv2D(dim, 3, padding='same')(m)
        n = BatchNormalization()(n) if bn else n
        n = Activation(acti)(n)
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, padding='same', dilation_rate=2)(n) if di else Conv2D(dim, 3, padding='same')(n)
        n = BatchNormalization()(n) if bn else n
        n = Activation(acti)(n)

        return n

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, di):
        if depth > 0:
            n = _conv_block(m, dim, acti, bn, di)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, di)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same')(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            n = Concatenate()([n, m])
            m = _conv_block(n, dim, acti, bn, di)
        else:
            m = _conv_block(m, dim, acti, bn, di, do)

        return m

    def _channel_attention(m, out_ch, ratio):
        avg_pool = GlobalAveragePooling2D()(m)
        avg_pool = Reshape((1,1,out_ch))(avg_pool)
        avg_pool = Dense(units=out_ch // ratio)(avg_pool)
        avg_pool = Activation('relu')(avg_pool)
        avg_pool = Dense(units=out_ch)(avg_pool)

        max_pool = GlobalMaxPooling2D()(m)
        max_pool = Reshape((1,1,out_ch))(max_pool)
        max_pool = Dense(units=out_ch // ratio)(max_pool)
        max_pool = Activation('relu')(max_pool)
        max_pool = Dense(units=out_ch)(max_pool)

        channel_attention = Add()([avg_pool, max_pool])
        channel_attention = Activation('sigmoid')(channel_attention)

        return multiply([m, channel_attention])

    def _spatial_attention(m, out_ch, ratio, kernel_size=7):
        if K.image_data_format() == "channels_first":
            spatial_attention = Permute((2,3,1))(m)
        else:
            spatial_attention = m

        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(spatial_attention)
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(spatial_attention)
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        spatial_attention = Conv2D(1, kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)

        if K.image_data_format() == "channels_first":
            spatial_attention = Permute((3,1,2))(spatial_attention)

        return multiply([m, spatial_attention])

    i = Input(shape=(None, None, input_channel_num))
    o = _level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual, dilated)
    o = Add()([o, _channel_attention(o, start_ch, ratio)]) if residual else _channel_attention(o, start_ch, ratio)
    o = Add()([o, _spatial_attention(o, start_ch, ratio)]) if residual else _channel_attention(o, start_ch, ratio)
    o = Conv2D(out_ch, 1)(o)
    model = Model(inputs=i, outputs=o)

    return model
