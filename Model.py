from keras.layers import Conv2D, BatchNormalization,Input
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
class Model():
    def __init__(self,options):
        self.options = options
        self.model = None

    def ResNet_Block(self,input,block_id,filter_num):
        x = BatchNormalization()(input)
        x = LeakyReLU(0.01)(x)

    def melody(self):
        input = Input(shape=(self.options.input_size, self.options.num_spec,1))
        
        block_1 = Conv2D(64,(3,3), name='conv1_1',padding='same',kernel_initializer='he_normal', use_bias=False,
                        kernel_regularizer=l2(1e-5))(input)
        block_1 = BatchNormalization()(block_1)
        block_1 = LeakyReLU(0.01)(block_1)
        block_1 = Conv2D(64,(3,3), name='conv1_2',padding='same', kernel_initializer='he_normal',use_bias=False,
                        kernel_regularizer=l2(1e-5))(block_1)