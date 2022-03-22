# The model is the DF model by Sirinam et al

from keras.models import Model
from keras.layers import Dense
from keras.layers import Input

from keras.layers import Activation
from keras.layers import ELU
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.core import Flatten
from keras.initializers import glorot_uniform
from keras.initializers import RandomNormal


def create_model(input_shape=None, emb_size=None, model_name=''):
    # -----------------Entry flow -----------------
    input_data = Input(shape=input_shape)

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]
    '''
    
    model1_out = Conv1D(filters=32, kernel_size=8, strides=1, padding='same')(input_data)
    model1_out = BatchNormalization(axis=-1)(model1_out)
    model1_out = ELU(alpha=1.0)(model1_out)
    model1_out = Conv1D(filters=32, kernel_size=8, strides=1, padding='same')(model1_out)
    model1_out = BatchNormalization(axis=-1)(model1_out)
    model1_out = ELU(alpha=1.0)(model1_out)
    model1_out = MaxPooling1D(pool_size=8, strides=4, padding='same')(model1_out)
    model1_out = Dropout(rate=0.1)(model1_out)

    model1_out = Conv1D(filters=64, kernel_size=8, strides=1, padding='same')(model1_out)
    model1_out = BatchNormalization()(model1_out)
    model1_out = Activation('relu')(model1_out)
    model1_out = Conv1D(filters=64, kernel_size=8, strides=1, padding='same')(model1_out)
    model1_out = BatchNormalization()(model1_out)
    model1_out = Activation('relu')(model1_out)
    model1_out = MaxPooling1D(pool_size=8, strides=4, padding='same')(model1_out)
    model1_out = Dropout(rate=0.1)(model1_out)

    print(model1_out._keras_shape)  # (None, 47, 64)
    model1_out = Flatten()(model1_out)
    # model1_out = Reshape((-1,))(model1_out)
    print(model1_out._keras_shape)  # (None, 3008)

    # Issue: OOM when allocating tensor with shape[650000,3000]
    # model1_out = Dense(1024, activation='relu', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(model1_out)
    # model1_out = Dropout(rate=0.6)(model1_out)
    model1_out = Dense(1024, kernel_initializer=glorot_uniform(seed=0))(model1_out)
    model1_out = BatchNormalization()(model1_out)
    model1_out = Activation('relu')(model1_out)
    model1_out = Dropout(rate=0.7)(model1_out)

    model1_out = Dense(1024, kernel_initializer=glorot_uniform(seed=0))(model1_out)
    model1_out = BatchNormalization()(model1_out)
    model1_out = Activation('relu')(model1_out)
    model1_out = Dropout(rate=0.5577112789569633)(model1_out)

    model1_out = Dense(1024, kernel_initializer=glorot_uniform(seed=0))(model1_out)
    model1_out = BatchNormalization()(model1_out)
    model1_out = Activation('sigmoid')(model1_out)
    model1_out = Dropout(rate=0.5)(model1_out)

    model1_out = Dense(emb_size, name='FeaturesVec')(model1_out)
    '''

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1'+'_'+model_name)(input_data)
    model = ELU(alpha=1.0, name='block1_adv_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv2'+'_'+model_name)(model)
    model = ELU(alpha=1.0, name='block1_adv_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                         padding='same', name='block1_pool'+'_'+model_name)(model)
    model = Dropout(0.1, name='block1_dropout'+'_'+model_name)(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1'+'_'+model_name)(model)
    model = Activation('relu', name='block2_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block2_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                         padding='same', name='block2_pool'+'_'+model_name)(model)
    model = Dropout(0.1, name='block2_dropout'+'_'+model_name)(model)

    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv1'+'_'+model_name)(model)
    model = Activation('relu', name='block3_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block3_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                         padding='same', name='block3_pool'+'_'+model_name)(model)
    model = Dropout(0.1, name='block3_dropout'+'_'+model_name)(model)

    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv1'+'_'+model_name)(model)
    model = Activation('relu', name='block4_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block4_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                         padding='same', name='block4_pool'+'_'+model_name)(model)

    output = Flatten()(model)

    dense_layer = Dense(emb_size, name='FeaturesVec'+'_'+model_name)(output)

    shared_conv2 = Model(inputs=input_data, outputs=dense_layer, name=model_name)
    return shared_conv2

def create_model_2d(input_shape=None, emb_size=None):
    input_data = Input(shape=input_shape) # (None, 2, 371, 1)
    # OOM when allocating tensor with shape[400,750,2,342]
    model = Conv2D(64, kernel_size=(2, 30), strides=(2, 1), padding='valid', activation='relu', input_shape=input_shape, kernel_initializer=RandomNormal(stddev=0.01))(input_data) #(None, 2, 342, 750)
    model = MaxPooling2D(pool_size=(1, 5), strides=(1, 1), padding='valid')(model) #(None, 2, 338, 2000)
    model = Conv2D(32, kernel_size=(1, 10), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=RandomNormal(stddev=0.01))(model) # (None, 2, 329, 1000)
    model = MaxPooling2D(pool_size=(1, 5), strides=(1, 1), padding='valid')(model) #(None, 2, 325, 1000)
    print(model._keras_shape) # (None, 2, 325, 1000)
    model = Flatten()(model)
    # model1_out = Reshape((-1,))(model1_out)
    print(model._keras_shape) # (None, 650000)
    # Issue: OOM when allocating tensor with shape[650000,3000]
    model = Dense(1024, activation='relu', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(model)
    model = Dropout(rate=0.6)(model)
    model = Dense(800, activation='relu', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(model)
    model = Dropout(rate=0.6)(model)
    model = Dense(100, activation='relu', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(model)
    model = Dropout(rate=0.6)(model)
    model = Dense(emb_size, activation='linear', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(model)
    shared_conv2 = Model(inputs=input_data, outputs=model)
    return shared_conv2