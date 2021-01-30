
from keras import losses
from keras.models import Model
from keras.layers import Input,  concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam


# PARAMETER INSTRUCTION RELATED TO SCALE FACTOR #
"""

SCALE 2:
        stride=5, inputsize=16

SCALE 3:
        stride=4, inputsize=11

SCALE 4:
        stride=3, inputsize=8

SCALE 8:
        stride=1, inputsize=4

"""

settings = \
{
"activation": "relu",
'augment':[], # must be any or all lof [90,180,270, 'flipud', 'fliplr', 'flipudlr' ]
'backend': 'tensorflow',
'batchsize':128,
'channels':1,
'colormode':'RGB', # 'YCbCr' or 'RGB'
'crop': 0,
'crop_test': 6,
'decay':1e-6,
'dilation_rate':(1,1),
'decimation': 'bicubic',
'espatience' : 50,
'epoch':50,
'inputsize':16, #
'interp_compare': 'lanczos',
'interp_up': 'bicubic',
'kernel_initializer': 'glorot_uniform',
'lrate':1e-3,
'lrpatience': 25,
'lrfactor' : 0.5,
'metrics': ["PSNR"],
'minimumlrate' : 1e-7,
'modelname':basename(__file__).split('.')[0],
'noise':'',
'normalization':['divide', '255.0'], # ['standard', "53.28741141", "40.73203139"],
'normalizeback': False,
'normalizeground':False,
'outputdir':'',
'scale':2,
'seed': 19,
'shuffle' : True,
'stride':5, # adımın 72 nin 1/3 ü olmasını istiyoruz. yani 24. 4 ölçek için 6 adım çıkışta 24 eder.
'target_channels': 1,
'target_cmode' : 'RGB',
'testpath': [r'D:\test'],
'traindir': r"D:\train",
'upscaleimage': False,
'valdir': r'D:\val',
'weightpath':'',
'workingdir': '',
}


# PARAMETERS FOR 2X SCALE
input_size = 16
lrate = 1e-3
decay = 1e-6
channels = 1

# INPUT 
input_shape = (input_size, input_size, channels)
main_input = Input(shape=input_shape, name='main_input')

# Feature extraction block (Lfeb)
L_FEB = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(main_input)
L_FEB = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(L_FEB)
L_FEB = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(L_FEB)
L_FEB = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(L_FEB)

# Feature upscaling layer (Lfup)
L_FUP = UpSampling2D(self.scale, name='upsampler_locally_connected')(L_FEB)

# Direct upscaling layer (Ldup)
L_DUP = UpSampling2D(self.scale)(main_input)

# REPEATING BLOCKS
RB1 = concatenate([L_FUP, L_DUP])
RB1 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB1)
RB1 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB1)
RB1 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB1)

RB2 = concatenate([L_FUP, L_DUP, RB1])
RB2 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB2)
RB2 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB2)
RB2 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB2)

RB3 = concatenate([L_FUP, L_DUP, RB1, RB2])
RB3 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB3)
RB3 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB3)
RB3 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB3)

RB4 = concatenate([L_FUP, L_DUP, RB1, RB2, RB3])
RB4 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB4)
RB4 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB4)
RB4 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB4)

# LAST LAYER
LAST = Conv2D(self.channels, (3, 3), kernel_initializer='glorot_uniform', activation='relu', padding='same')(RB4)

model = Model(main_input, outputs=LAST)
model.compile(Adam(lrate, decay), loss=losses.mean_squared_error)
