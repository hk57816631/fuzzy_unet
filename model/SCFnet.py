from keras.models import Model,model_from_json
from keras.layers import Input, multiply, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, subtract, add
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.core import Lambda
import tensorflow as tf
import math
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def log2(x, class_num):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(class_num, dtype=numerator.dtype))
  return numerator / denominator

def fuzzyfunc(x):
    fuzzy_W_tensor1 = tf.nn.relu(tf.subtract(0.5, x))
    fuzzy_W_tensor2 = tf.nn.relu(tf.subtract(x, 0.5))
    # map: 0--0.5 => 0--0.5; 0.5--1 => 0.5--0
    fuzzy_W_tensor = tf.subtract(0.5, tf.add(fuzzy_W_tensor1, fuzzy_W_tensor2))
    # map to 0--1
    fuzzy_W_tensor = tf.multiply(2.0, fuzzy_W_tensor)
    fuzzy_W_tensor = tf.subtract(1.0, fuzzy_W_tensor)
    return fuzzy_W_tensor
def axismax(x):
    x_max = tf.reduce_max(x, axis=3)
    x_max = tf.reshape(x_max, [-1, x_max.shape[1].value, x_max.shape[2].value, 1])
    return x_max

def info_entropy(x, class_num):
    # c = 0
    # nclass = 2
    # membership = Lambda(lambda x: tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 1]))(x)
    # med = tf.multiply(membership, log2(tf.add(membership, 1e-10)))
    # entro = tf.subtract(0.0, med)
    # for i in range(nclass):
    #     c = c-1/nclass * math.log2(1/nclass)
    # for i in range(1, nclass):
    #     membership = Lambda(lambda x: tf.slice(x, [0, 0, 0, i], [-1, -1, -1, 1]))(x)
    #     med = tf.multiply(membership, log2(tf.add(membership, 1e-10)))
    #     entro = tf.subtract(entro, med)
    # return tf.divide(entro, c)
    #tmp = log2(x)
    result = tf.subtract(0.0, tf.reduce_sum(tf.multiply(x, log2(x, class_num)), axis=-1))
    result = tf.expand_dims(result, axis=-1)
    return result
def get_certain(x):
    return tf.subtract(1.0, x)
def define_softmax(x):
    output = tf.nn.softmax(x, axis=-1)
    return output
def define_l1(x):
    output = tf.reduce_sum(x, axis = -1)
    output = tf.divide(x, output)
    return output
def channel_fuzzy(x):
    xc = GlobalAveragePooling2D()(x)
    xc = Dense(256, activation='relu')(xc)
    #xc = Dense(128, activation='relu')(xc)
    result = Dense(x.shape[3].value, activation='sigmoid')(xc)
    return result
def fuzzy_block(x, num_classes, x_pre):
    conv1 = Conv2D(16, (1, 1), padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(16, (1, 1), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    fuzzyLayer = Conv2D(num_classes, (1, 1), padding='same')(conv1)
    fuzzyLayer = BatchNormalization()(fuzzyLayer)
    fuzzyLayer = Lambda(lambda x: define_softmax(x))(fuzzyLayer)
    fuzzy_feature = Conv2D(x.shape[3].value, (3, 3), padding='same')(x_pre)
    fuzzy_feature = BatchNormalization()(fuzzy_feature)
    ############### DSC Feature
    # conv1_dsn1 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x_pre)
    # conv2_dsn1 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(conv1_dsn1)
    #
    # context_conv1_1 = Conv2D(filters=32, kernel_size=1)(conv2_dsn1)
    # # context2_1 = iRNN()(block_1_out)
    # context1_1 = iRNN()(context_conv1_1)
    # atten_c1_context1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(
    #     x_pre)
    # atten_c2_context1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(
    #     atten_c1_context1)
    # atten_c3_context1 = Conv2D(filters=128, kernel_size=1, activation='sigmoid')(
    #     atten_c2_context1)
    # #     (1, 256, 256, 128)
    # #     (1, 256, 256, 128)
    # context1_1a = multiply([context1_1, atten_c3_context1])
    # context1 = Conv2D(filters=128, padding='same', kernel_size=1, activation='relu')(context1_1a)
    # dsn1_feture = concatenate([context1, conv2_dsn1])
    # conv3_dsn1 = Conv2D(filters=x.shape[3].value, kernel_size=1)(dsn1_feture)
    ###########################
    uncertain = Lambda(lambda x: info_entropy(x, num_classes))(fuzzyLayer)
    certain = Lambda(lambda x: get_certain(x))(uncertain)

    #fuzzy_or = Lambda(lambda x: axismax(x))(uncertain)
    result = add([multiply([x, certain]), multiply([fuzzy_feature, uncertain])])
    result = Activation('relu')(result)
    #result = multiply([x, certain ])
    #result = Activation('relu')(result)
    return result


def SCFnet(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    img_input = Input(input_shape)
    #fuzzy_input = fuzzy_block(img_input, num_classes)
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    block_1_out = Activation('relu')(x)
    fuzzy_1_out = fuzzy_block(block_1_out, num_classes, img_input)

    b2 = x = MaxPooling2D()(fuzzy_1_out)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    block_2_out = Activation('relu')(x)
    fuzzy_2_out = fuzzy_block(block_2_out, num_classes, b2)

    b3 = x = MaxPooling2D()(fuzzy_2_out)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    block_3_out = Activation('relu')(x)
    fuzzy_3_out = fuzzy_block(block_3_out, num_classes, b3)

    b4 = x = MaxPooling2D()(fuzzy_3_out)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    block_4_out = Activation('relu')(x)
    fuzzy_4_out = fuzzy_block(block_4_out, num_classes, b4)

    b5 = x = MaxPooling2D()(fuzzy_4_out)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = fuzzy_block(x, num_classes, b5)

    for_pretrained_weight = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, for_pretrained_weight)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, fuzzy_4_out])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    c = channel_fuzzy(x)
    x = multiply([x, c])
    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, fuzzy_3_out])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    c = channel_fuzzy(x)
    x = multiply([x, c])

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, fuzzy_2_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    c = channel_fuzzy(x)
    x = multiply([x, c])

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, fuzzy_1_out])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    c = channel_fuzzy(x)
    x = multiply([x, c])

    # last conv
    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    return model

def net_from_json(path, lr_init, lr_decay):
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),loss='categorical_crossentropy', metrics=[dice_coef])

    return model
