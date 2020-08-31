from keras.layers import Conv2D, BatchNormalization,Input , MaxPooling2D,add,Reshape,\
                            Bidirectional,LSTM, TimeDistributed,concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers.core import Dropout, Dense , Activation

import math

class Model():
    def __init__(self,options):
        self.options = options
        self.model = None
    
    def jdc_network(self):
        self.main_network()

    def resnet_block(self,input,block_id,filter_num):
        x = BatchNormalization()(input)
        x = LeakyReLU(0.01)(x)
        x = MaxPooling2D((1,4))(x)

        skip = Conv2D(filter_num, (1,1), name='conv'+str(block_id)+'1x1',padding='same',kernel_initializer='he_normal',use_bias=False)(x)

        x = Conv2D(filter_num, (3,3) , name='conv'+str(block_id)+'_1',padding='same',kernel_initializer='he_normal',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.01)(x)
        x = Conv2D(filter_num, (3,3), name='conv'+str(block_id)+'_2',padding='same',kernel_initializer='he_normal',use_bias=False)(x) 

        return add([skip,x])
           
    def main_network(self):
        input = Input(shape=(self.options.input_size, self.options.num_spec,1))
        
        block_1 = Conv2D(64,(3,3), name='conv1_1',padding='same',kernel_initializer='he_normal', use_bias=False,
                        kernel_regularizer=l2(1e-5))(input)
        block_1 = BatchNormalization()(block_1)
        block_1 = LeakyReLU(0.01)(block_1)
        block_1 = Conv2D(64,(3,3), name='conv1_2',padding='same', kernel_initializer='he_normal',use_bias=False,
                        kernel_regularizer=l2(1e-5))(block_1)

        block_2 = self.resnet_block(input=block_1, block_id=2, filter_num=128)
        block_3 = self.resnet_block(input=block_2, block_id=3, filter_num=192)
        block_4 = self.resnet_block(input=block_3, block_id=4, filter_num=256)

        block_4 = BatchNormalization()(block_4)
        block_4 = LeakyReLU(0.01)()(block_4)
        block_4 = MaxPooling2D((1,4))(block_4)
        block_4 = Dropout(0.5)(block_4)

        output = Reshape((self.options.input_size, block_4.shape[2] * block_4.shape[3]))(block_4)
        output = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.3, dropout=0.3))(output)

        num_output = int(45 * 2 ** (math.log(self.options.resolution, 2)) + 2)
        output = TimeDistributed(Dense(num_output))(output)
        output = TimeDistributed(Activation("softmax"), name='output')(output)

        return block_1, block_2, block_3, output
    
    def auxiliary_network(self,b_1, b_2, b_3 , b_4):
        block_1 = MaxPooling2D((1, 4 ** 4))(b_1)
        block_2 = MaxPooling2D((1, 4 ** 3))(b_2)
        block_3 = MaxPooling2D((1, 4 ** 2))(b_3)
        block_4 = b_4

        joint = concatenate([block_1,block_2,block_3,block_4])
        joint = Conv2D(256, (1,1), padding='same', kernel_initializer='he_normal', use_bias=False, 
                        kernel_regularizer=l2(1e-5))(joint)
        joint = BatchNormalization()(joint)
        joint = LeakyReLU(0.01)(joint)
        joint = Dropout(0.5)(joint)

        output = Reshape((self.options.input_size, joint.shape[2] * joint.shape[3]))(joint)
        output = Bidirectional(LSTM(32,return_sequences=True, stateful=False, recurrent_dropout=0.3, dropout=0.3))(output)
        output = TimeDistributed(Dense(2))(output)
        output = TimeDistributed(Activation("softmax"))(output)