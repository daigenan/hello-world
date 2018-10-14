hell
class Factory(object):
    def get_model(self, conf, arm_shape):
        print "use model ", conf.model_name
        model = None
        function_name = "{}_model(conf, arm_shape)".format(conf.model_name)
        exec "model = self." + function_name
        return model

    def RNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = MyReshape(conf.batch_size)(input_x)
        # output = SimpleRNN(32, return_sequences=True)(output)
        output = SimpleRNN(conf.observe_length)(output)
        # output = Dropout(0.1)(output)
        output = Dense(conf.predict_length, activation="tanh")(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model

    def LSTM_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = MyReshape(conf.batch_size)(input_x)
        output = LSTM(conf.observe_length)(output)
        output = Dense(conf.predict_length)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=input_x, outputs=output)
        return model


    def CRNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        input_x = Input((road_num, conf.observe_length, 1))
        output = Conv2D(32, (2, 2), strides=(1, 1), padding="same")(input_x)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Activation(activation="relu")(output)
        output = Conv2D(16, (2, 2), strides=(1, 1), padding="same")(output)
        # pool2 = AveragePooling2D(pool_size=(1,2))(conv2)
        # pool2 = Activation(activation="sigmoid")(conv2)
        # conv3 = Conv2D(4, (2, 2), strides=(1, 1), padding="same")(pool2)
        # pool3 = AveragePooling2D(pool_size=(1, 2))(conv3)
        output = Activation(activation="relu")(output)
        output = MyReshape(conf.batch_size)(output)
        output = SimpleRNN(5)(output)
        output = Dense(conf.predict_length)(output)
        output = MyInverseReshape(conf.batch_size)(output)
        # f = Flatten()(pool3)
        # output = Dense(road_num * conf.predict_length, activation="sigmoid")(f)
        # output = Reshape((road_num, conf.predict_length))(output)
        model = Model(inputs=input_x, outputs=output)
        return model


    def LCRNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)

        output = MyReshape(conf.batch_size)(output)
        output = SimpleRNN(5)(output)
        inputs = [input_x, input_ram]

        if conf.use_externel:
            output = Dense(conf.predict_length, activation="relu")(output)
            output = MyInverseReshape(conf.batch_size)(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                outputs.append(matrixLayer()(output_e))
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation("tanh")(output)
        else:
            output = Dense(conf.predict_length, activation="tanh")(output)
            output = MyInverseReshape(conf.batch_size)(output)

        model = Model(inputs=inputs, outputs=output)
        return model

    def __E_input_output(self, conf, arm_shape, activation="tanh"):
        road_num = arm_shape[0]
        if conf.observe_p != 0:
            input_x1 = Input((road_num, conf.observe_p))
            output1 = MyReshape(conf.batch_size)(input_x1)
            output1 = Dense(conf.observe_p + 1, activation="relu")(output1)

        if conf.observe_t != 0:
            input_x2 = Input((road_num, conf.observe_t))
            output2 = MyReshape(conf.batch_size)(input_x2)
            output2 = Dense(conf.observe_t + 1, activation="relu")(output2)

        if conf.observe_p != 0:
            if conf.observe_t != 0:
                output = Concatenate()([output1, output2])
                input_x = [input_x1, input_x2]
            else:
                output = output1
                input_x = input_x1
        else:
            output = output2
            input_x = input_x2

        output = Dense(conf.predict_length, activation=activation)(output)
        output = MyInverseReshape(conf.batch_size)(output)

        input_x3 = Input((conf.predict_length, 37))  # 37 is externel dim
        if isinstance(input_x, list):
            input_x += [input_x3]
        else:
            input_x = [input_x, input_x3]

        output_3 = MyReshape(conf.batch_size)(input_x3)
        output_3 = Dense(road_num, activation=activation)(output_3)
        output_3 = MyInverseReshape(conf.batch_size)(output_3)
        output_3 = Reshape((road_num, conf.predict_length))(output_3)
        output = Add()([output, output_3])
        return input_x, output

    def E_model(self, conf, arm_shape):
        input_x, output = self.__E_input_output(conf, arm_shape)
        model = Model(inputs=input_x, output=output)
        return model

    def LCRNNBN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = MyReshape(conf.batch_size)(output)
        output = SimpleRNN(5)(output)
        inputs = [input_x, input_ram]
        if conf.use_externel:
            output = Dense(conf.predict_length, activation="relu")(output)
            output = MyInverseReshape(conf.batch_size)(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                outputs.append(matrixLayer()(output_e))
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation("tanh")(output)
        else:
            output = Dense(conf.predict_length, activation="tanh")(output)
            output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=inputs, outputs=output)
        return model

    def LCNN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        # input_effective = Input((arm_shape[0],))
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)
        # output = Effective()([output, input_effective])
        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)
        # output = Effective()([output, input_effective])

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = LookUpSqueeze()(output)
        inputs = [input_x, input_ram]

        if conf.use_externel:
            output = Conv2D(1, (1, 5), activation="relu")(output)
            output = Reshape((road_num, conf.predict_length))(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                outputs.append(matrixLayer()(output_e))
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation("tanh")(output)
        else:
            output = Conv2D(1, (1, 5), activation="tanh")(output)
            output = Reshape((road_num, conf.predict_length))(output)

        model = Model(inputs=inputs, outputs=output)
        return model

    def LCLSTMBN_model(self, conf, arm_shape):
        road_num = arm_shape[0]
        A = arm_shape[1]
        input_x = Input((road_num, conf.observe_length, 1))
        input_ram = Input(arm_shape)
        output = Lookup(conf.batch_size)([input_x, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = Lookup(conf.batch_size)([output, input_ram])
        output = Conv3D(16, (1, A, 2), activation="relu")(output)
        output = BatchNormalization()(output)
        output = LookUpSqueeze()(output)

        output = MyReshape(conf.batch_size)(output)
        output = LSTM(5)(output)
        inputs = [input_x, input_ram]
        if conf.use_externel:
            output = Dense(conf.predict_length, activation="relu")(output)
            output = MyInverseReshape(conf.batch_size)(output)
            input_e, output_e = self.__E_input_output(conf, arm_shape)
            if isinstance(input_e, list):
                inputs += input_e
            else:
                inputs += [input_e]
            if conf.use_matrix_fuse:
                outputs = [matrixLayer()(output)]
                outputs.append(matrixLayer()(output_e))
                output = Add()(outputs)
            else:
                output = Add()([output, output_e])
            output = Activation("tanh")(output)
        else:
            output = Dense(conf.predict_length, activation="tanh")(output)
            output = MyInverseReshape(conf.batch_size)(output)
        model = Model(inputs=inputs, outputs=output)
        return model

class Lookup(Layer):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        super(Lookup, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        arm = inputs[1]
        if K.dtype(arm) != 'int32':
            arm = K.cast(arm, 'int32')

        outs = []
        for _i in range(self.batch_size):
            out1 = tf.nn.embedding_lookup(x[_i], arm[_i])  #return all road SX matrices?
            outs.append(out1)
        out = tf.stack(outs, axis=0)
        return out

    def compute_output_shape(self, input_shape):
        x_shape = input_shape[0]
        arm_shape = input_shape[1]
        f_num = x_shape[3]
        r_num = x_shape[1]
        t_num = x_shape[2]
        a_num = arm_shape[2]

        return (x_shape[0], r_num, a_num, t_num, f_num)


class LookUpSqueeze(Layer):
    def call(self, inputs, **kwargs):
        output = tf.squeeze(inputs, axis=2)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + input_shape[3:]


class MyReshape(Layer):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        super(MyReshape, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_shape = inputs.get_shape().as_list()
        return tf.reshape(inputs, (self.batch_size * input_shape[1],) + tuple(input_shape[2:]))

    def compute_output_shape(self, input_shape):
        return (self.batch_size * input_shape[1],) + tuple(input_shape[2:])


class MyInverseReshape(Layer):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        super(MyInverseReshape, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_shape = inputs.get_shape().as_list()
        return tf.reshape(inputs, (self.batch_size, input_shape[0] / self.batch_size, input_shape[1]))

    def compute_output_shape(self, input_shape):
        return (self.batch_size, input_shape[0] / self.batch_size, input_shape[1])


class matrixLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(matrixLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape


