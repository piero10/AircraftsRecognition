# -*- coding: utf-8 -*-
'''Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790.
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

Also do note that this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers.

# Reference:

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

'''
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Activation, merge
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras import backend as K
from imagenet_utils import decode_predictions


TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Xception(include_top=True, weights='imagenet',
             input_tensor=None):
    '''Instantiate the Xception architecture,
    optionally loading weights pre-trained
    on ImageNet. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    dimension ordering `(width, height, channels)`.
    You should set `image_dim_ordering="tf"` in your Keras config
    located at ~/.keras/keras.json.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    if K.backend() != 'tensorflow':
        raise Exception('The Xception model is only available with '
                        'the TensorFlow backend.')
    if K.image_dim_ordering() != 'tf':
        warnings.warn('The Xception model is only available for the '
                      'input dimension ordering "tf" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'dimension ordering "th" (channels, width, height). '
                      'You should set `image_dim_ordering="tf"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "tf" dimension ordering.')
        K.set_image_dim_ordering('tf')
        old_dim_ordering = 'th'
    else:
        old_dim_ordering = None

    # Determine proper input shape
    if include_top:
        input_shape = (299, 299, 3)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(32, 3, 3, subsample=(2, 2), bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, 3, 3, bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block2_pool')(x)
    x = merge([x, residual], mode='sum')

    residual = Conv2D(256, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block3_pool')(x)
    x = merge([x, residual], mode='sum')

    residual = Conv2D(728, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block4_pool')(x)
    x = merge([x, residual], mode='sum')

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = merge([x, residual], mode='sum')

    residual = Conv2D(1024, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block13_pool')(x)
    x = merge([x, residual], mode='sum')

    x = SeparableConv2D(1536, 3, 3, border_mode='same', bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, 3, 3, border_mode='same', bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)

    if old_dim_ordering:
        K.set_image_dim_ordering(old_dim_ordering)
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = Xception(include_top=True, weights='imagenet')

    img_path = 'f15.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, 1))
