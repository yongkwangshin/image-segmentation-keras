import argparse
import Models , LoadBatches



parser = argparse.ArgumentParser() #parser이라는 객체를 만든다.
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int ) # "--n_classes ~" 얘의 값을 parser.parse_args.n_classes라는 항목으로 만든다.
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args() # parser.parse_args.n_classes 를 짧게 args.n_classes 쓰려고 객체 args 만듬
# 기본값들을 설정해준다.
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
# 최적화 방법과 모델 이름을 설정한다.
optimizer_name = args.optimizer_name
model_name = args.model_name

if validate: # validata(검증 데이터)가 있으면 검증데이터도 설정해준다.
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ] # 위에 모델 중에 원하는 모델을 modelFN에 넣는다, 상황에 따라 원하는 모델로 바꿀 수 있으므로 편리하다. (python 자료형인 딕셔너리 알아야 위에 코드 이해가능)

m = modelFN( n_classes , input_height=input_height, input_width=input_width   ) # 모델을 m에 장착해준다. h,w는 입력 기본값이 이미 함수안에 있어서 입력 안해도됨 (Models 폴더에서 하나 들어가서 데이터 입력형태 보면 이해가능)
m.compile(loss='categorical_crossentropy',# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      optimizer= optimizer_name ,
      metrics=['accuracy']) #  m.compile = keras.models.Sequential.compile 임 (Segnet 보면 이렇게 나옴)
# 윗줄의 의미 : lossfunc은 loss를, optimizer는 optimizer를 정확도 측정 파라미터는 metrics를 사용한다.
# https://keras.io/models/sequential/ 케라스 공식 문서,가이드
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
		m.fit_generator( G , 512  , epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )
else:
	for ep in range( epochs ):
		m.fit_generator( G , 512  , validation_data=G2 , validation_steps=200 ,  epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep )  )
		m.save( save_weights_path + ".model." + str( ep ) )


