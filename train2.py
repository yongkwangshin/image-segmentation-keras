import argparse
import Models , LoadBatches



parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 1 )
parser.add_argument("--batch_size", type = int, default = 5 )
parser.add_argument("--val_batch_size", type = int, default = 5 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adam")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

# VGGSegnet model
import tensorflow as tf
import os
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"/data/vgg16_weights_th_dim_ordering_th_kernels.h5"

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

m = VGGSegnet( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='binary_crossentropy',
      optimizer= optimizer_name ,
      metrics=['accuracy'])


if len( load_weights ) > 0:
	m.load_weights(load_weights)


print "Model output shape" ,  m.output_shape

output_height = m.outputHeight
output_width = m.outputWidth

G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


if validate:
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 30  , epochs=5 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )
else:
	for ep in range( epochs ):
		m.fit_generator( G , 30  , validation_data=G2 , validation_steps=50 ,  epochs=5 )
		m.save_weights( save_weights_path + "." + str( ep )  )
		m.save( save_weights_path + ".model." + str( ep ) )