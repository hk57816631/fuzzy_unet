from keras.models import Model
from keras.layers import Input, multiply, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.core import Lambda
from model.gaussLayer import FuzzyLayer, FuzzyLayer_sigmoid
import tensorflow as tf

# compute the dice-coefficient
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

# mapping fuzzy membership to uncertainty map
def fuzzyfunc(x):
    fuzzy_W_tensor1 = tf.nn.relu(tf.subtract(0.5, x))
    fuzzy_W_tensor2 = tf.nn.relu(tf.subtract(x, 0.5))
    # map: 0--0.5 => 0--0.5; 0.5--1 => 0.5--0
    fuzzy_W_tensor = tf.subtract(0.5, tf.add(fuzzy_W_tensor1, fuzzy_W_tensor2))
    # map to 0--1
    fuzzy_W_tensor = tf.multiply(2.0, fuzzy_W_tensor)
    fuzzy_W_tensor = tf.subtract(1.0, fuzzy_W_tensor)
    return fuzzy_W_tensor

# fuzzy AND operation
def axismax(x):
    x_max = tf.reduce_max(x, axis=4)
    return x_max

# network structure
def fuzzy_unet(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    # input layer
    img_input = Input(input_shape)

    # ============================================================================
    # first fuzzy block
    # fuzzification
    fuzzyLayer1 = FuzzyLayer_sigmoid((input_shape[0], input_shape[1], input_shape[2]),
                            (input_shape[0], input_shape[1], input_shape[2], num_classes))(img_input)
    # uncertainty map
    fuzzy_W_tensor1 = Lambda(lambda x: fuzzyfunc(x))(fuzzyLayer1)
    # fuzzy AND
    x_max1 = Lambda(axismax)(fuzzy_W_tensor1)
    # fusion
    productlayer1 = multiply([img_input, x_max1])
    # ============================================================================

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(productlayer1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    block_1_out = Activation('relu')(x)

    # ============================================================================
    # second fuzzy block
    # fuzzification
    fuzzyLayer2 = FuzzyLayer_sigmoid((block_1_out.shape[1].value, block_1_out.shape[2].value, block_1_out.shape[3].value),
                            (block_1_out.shape[1].value, block_1_out.shape[2].value, block_1_out.shape[3].value, num_classes))(block_1_out)
    # uncertainty map
    fuzzy_W_tensor2 = Lambda(lambda x: fuzzyfunc(x))(fuzzyLayer2)
    # fuzzy AND
    x_max2 = Lambda(axismax)(fuzzy_W_tensor2)
    # fusion
    productlayer2 = multiply([block_1_out, x_max2])
    # ============================================================================

    x = MaxPooling2D()(productlayer2)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    block_2_out = Activation('relu')(x)

    x = MaxPooling2D()(block_2_out)

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

    x = MaxPooling2D()(block_3_out)

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

    x = MaxPooling2D()(block_4_out)

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

    for_pretrained_weight = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, for_pretrained_weight)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_4_out])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_3_out])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_2_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x, block_1_out])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # last conv
    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    return model


