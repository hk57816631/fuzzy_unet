# the defined fuzzification layer; two membership functions are used: Gaussian and Sigmoid

from keras.layers import Layer
import tensorflow as tf

# fuzzification layer with Gaussian membership function
class FuzzyLayer(Layer):
    # initialization: input dimension; output dimension
    def __init__(self, input_dim, output_dim, **kwargs):
        self.img_height = input_dim[0]
        self.img_width = input_dim[1]
        self.channel_num = input_dim[2]
        self.class_num = output_dim[3]
        self.output_dim = output_dim
        super(FuzzyLayer, self).__init__(**kwargs)

    # define variables in this layer: mean: the mean value of the fuzzy membership function
    #                                 scale: the variance of the fuzzy membership function
    def build(self, input_shape):
        # initialization: uniform random distribution; trainable variable
        self.scale = self.add_weight(name='v1', shape=(self.class_num, self.img_height, self.img_width, self.channel_num),
                                     initializer='uniform', trainable=True)
        # initialization: uniform random distribution; trainable variable
        self.mean = self.add_weight(name='v2', shape=(self.class_num, self.img_height, self.img_width, self.channel_num),
                                    initializer='uniform', trainable=True)
        super(FuzzyLayer, self).build(input_shape)

    # define the callable function in this layer: Gaussian function
    def call(self, x):
        output = []
        # compute Gaussian function for each category i, just depend on the defination of Gaussian function
        for i in range(self.class_num):
            # subtract mean
            x1 = tf.subtract(x, self.mean[i])
            # compute square after mean
            x2 = tf.square(x1)
            # divide variance
            x3 = tf.divide(x2, self.scale[i])
            # multiply -1/2
            x4 = tf.multiply(-0.5, x3)
            # exponential function
            gauss = tf.exp(x4)

            output.append(gauss)

        # convert output from tuple to tensor
        output = tf.convert_to_tensor(output)
        # transpose the tensor to be the same dimension as input
        output = tf.transpose(output, perm=[1, 2, 3, 4, 0])
        # L1 normalize. make sure summation of fuzzy membership of each pixel in each category equals to 1
        # compute the sum of fuzzy membership of each pixel in each category
        sum1 = tf.reduce_sum(output, axis=4)
        # tile the shape to the same size as output, i.e. make sure all the pixels have number of sum as the category
        # number
        sum1 = tf.reshape(sum1, [-1, self.img_height, self.img_width, self.channel_num, 1])
        sum1 = tf.tile(sum1, multiples=[1, 1, 1, 1, self.class_num])
        # L1 normalization
        output = tf.divide(output, sum1)
        return output

    # set the output size
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], self.output_dim[2], self.class_num)
        return output_shape


# fuzzification layer with Sigmoid membership function
class FuzzyLayer_sigmoid(Layer):
    # initialization: input dimension; output dimension
    def __init__(self, input_dim, output_dim, **kwargs):
        self.img_height = input_dim[0]
        self.img_width = input_dim[1]
        self.channel_num = input_dim[2]
        self.class_num = output_dim[3]
        self.output_dim = output_dim
        super(FuzzyLayer_sigmoid, self).__init__(**kwargs)

    # define variables in this layer: a: steeper degree of Sigmoid function
    #                                 b: the center location of Sigmoid function
    def build(self, input_shape):
        # initialization: uniform random distribution; trainable variable
        self.a = self.add_weight(name='v1',
                                     shape=(self.class_num, self.img_height, self.img_width, self.channel_num),
                                     initializer='uniform', trainable=True)
        # initialization: uniform random distribution; trainable variable
        self.b = self.add_weight(name='v2',
                                    shape=(self.class_num, self.img_height, self.img_width, self.channel_num),
                                    initializer='uniform', trainable=True)
        super(FuzzyLayer_sigmoid, self).build(input_shape)

    # define the callable function in this layer: Sigmoid function
    def call(self, x):
        output = []
        # compute Gaussian function for each category i, just depend on the defination of Sigmoid function
        for i in range(self.class_num):
            x1 = tf.subtract(x, self.b[i])
            x2 = tf.multiply(self.a[i], x1)
            gauss = tf.nn.sigmoid(x2)
            output.append(gauss)
        # convert output from tuple to tensor
        output = tf.convert_to_tensor(output)
        # transpose the tensor to be the same dimension as input
        output = tf.transpose(output, perm=[1, 2, 3, 4, 0])
        # L1 normalize. make sure summation of fuzzy membership of each pixel in each category equals to 1
        # compute the sum of fuzzy membership of each pixel in each category
        sum1 = tf.reduce_sum(output, axis=4)
        # tile the shape to the same size as output, i.e. make sure all the pixels have number of sum as the category
        # number
        sum1 = tf.reshape(sum1, [-1, self.img_height, self.img_width, self.channel_num, 1])
        sum1 = tf.tile(sum1, multiples=[1, 1, 1, 1, self.class_num])
        # L1 normalization
        output = tf.divide(output, sum1)
        return output

    # set the output size
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim[0], self.output_dim[1], self.output_dim[2], self.class_num)
        return output_shape

