
from keras import losses
from keras.models import Model
from keras.layers import Input,  concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam

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
