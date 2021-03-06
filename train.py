import argparse
import Models , LoadBatches
import tensorflow as tf
import h5py

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

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32  }
modelFN = modelFns[ model_name ]
opt = tf.keras.optimizers.Adam(lr=0.001)
m = Models.VGGSegnet.VGGSegnet( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='binary_crossentropy',
      optimizer= opt ,
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
		m.fit_generator( G , 100  , epochs=30 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )
else:
	for ep in range( epochs ):
		m.fit_generator( G , 10  , validation_data=G2 , validation_steps=50 ,  epochs=2 )
		# m.save_weights( save_weights_path + "." + str( ep )  )
		# m.save( save_weights_path + ".model." + str( ep ) )
		tf.keras.models.save_model(m,save_weights_path + ".model." + str( ep ))
		# tf.keras.models.Model.save(m,save_weights_path + ".model." + str( ep ))
		# model_json = tf.keras.models.to_json()
		# with open("model.json", "w") as json_file : 
		# json_file.write(model_json)

		
		# tf.keras.models.save_model(m,save_weights_path + ".model." + str( ep ))


