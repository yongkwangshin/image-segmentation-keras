



import tensorflow as tf
from keras.models import *

import os
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"


def VGGSegnet( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

	vgg_level=2
	img_input = tf.keras.Input(shape=(3,input_height,input_width))

	x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first' )(img_input)
	x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first' )(x)
	x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first' )(x)
	f1 = x
	# Block 2
	x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_first' )(x)
	x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_first' )(x)
	x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first' )(x)
	f2 = x

	# Block 3
	x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_first' )(x)
	x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_first' )(x)
	x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_first' )(x)
	x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_first' )(x)
	f3 = x

	# Block 4
	x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_first' )(x)
	x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_first' )(x)
	x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_first' )(x)
	x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_first' )(x)
	f4 = x

	# Block 5
	x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_first' )(x)
	x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_first' )(x)
	x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_first' )(x)
	x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_first' )(x)
	f5 = x

	x = tf.keras.layers.Flatten(name='flatten')(x)
	x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
	x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
	x = tf.keras.layers.Dense( 1000 , activation='softmax', name='predictions')(x)

	vgg  = tf.keras.Model(  img_input , x  )
	vgg.load_weights(VGG_Weights_path)

	levels = [f1 , f2 , f3 , f4 , f5 ]

	o = levels[ vgg_level ]
	
	o = ( tf.keras.layers.ZeroPadding2D( (1,1) , data_format='channels_first' ))(o)
	o = ( tf.keras.layers.Conv2D(32, (3, 3), padding='valid', data_format='channels_first'))(o)
	o = ( tf.keras.layers.BatchNormalization())(o)

	o = ( tf.keras.layers.UpSampling2D( (2,2), data_format='channels_first'))(o)
	o = ( tf.keras.layers.ZeroPadding2D( (1,1), data_format='channels_first'))(o)
	o = ( tf.keras.layers.Conv2D( 16, (3, 3), padding='valid', data_format='channels_first'))(o)
	o = ( tf.keras.layers.BatchNormalization())(o)

	o = ( tf.keras.layers.UpSampling2D((2,2)  , data_format='channels_first' ) )(o)
	o = ( tf.keras.layers.ZeroPadding2D((1,1) , data_format='channels_first' ))(o)
	o = ( tf.keras.layers.Conv2D( 8 , (3, 3), padding='valid' , data_format='channels_first' ))(o)
	o = ( tf.keras.layers.BatchNormalization())(o)

	o = ( tf.keras.layers.UpSampling2D((2,2)  , data_format='channels_first' ))(o)
	o = ( tf.keras.layers.ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
	o = ( tf.keras.layers.Conv2D( 4 , (3, 3), padding='valid'  , data_format='channels_first' ))(o)
	o = ( tf.keras.layers.BatchNormalization())(o)


	o =  tf.keras.layers.Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_first' )( o )
	o_shape = tf.keras.Model(img_input , o ).output_shape
	outputHeight = o_shape[2]
	outputWidth = o_shape[3]

	o = (tf.keras.layers.Reshape((  -1  , outputHeight*outputWidth   )))(o)
	o = (tf.keras.layers.Permute((2, 1)))(o)
	o = (tf.keras.layers.Activation('sigmoid'))(o)
	model = tf.keras.Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	return model




if __name__ == '__main__':
	m = VGGSegnet( 101 )
	tf.keras.utils.plot_model( m , show_shapes=True , to_file='model.png')

