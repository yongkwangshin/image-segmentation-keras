
# todo upgrade to keras 2.0


from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Merge
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam , SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
# from keras.regularizers import ActivityRegularizer
from keras import backend as K





def segnet(nClasses , optimizer=None , input_height=360, input_width=480 ):

	kernel = 3 # 필터 = 커널
	filter_size = 64 
	pad = 1 # 패딩 사이즈
	pool_size = 2 # 풀링 사이즈

	model = models.Sequential() # 모델을 연속적으로 쌓아가겠다걸 알리는 함수, 별의미 없음
	model.add(Layer(input_shape=(3, input_height , input_width ))) # add 함수 : layer 쌓는 함수

	# encoder
	model.add(ZeroPadding2D(padding=(pad,pad))) # (1,1)로 zeropadding 한다.
	model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid')) # 왜 커널이 두개 있는지 모르겠지만 아무튼 컨볼루션 layer
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
# 제로패딩, 컨볼루션, 정규화, Relu, 맥스풀링을 반복한다.
	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(128, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
# 제로패딩, 컨볼루션, 정규화, Relu, 맥스풀링을 반복한다.
	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(256, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
# 제로패딩, 컨볼루션, 정규화, Relu, 맥스풀링을 반복한다.
	model.add(ZeroPadding2D(padding=(pad,pad)))
	model.add(Convolution2D(512, kernel, kernel, border_mode='valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
# 마지막단은 맥스풀링할 필요없다.

	# decoder
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(512, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())
# 다운 샘플링 할때처럼 여기서는 업샘플링 반복한다.
	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(256, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(128, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())

	model.add( UpSampling2D(size=(pool_size,pool_size)))
	model.add( ZeroPadding2D(padding=(pad,pad)))
	model.add( Convolution2D(filter_size, kernel, kernel, border_mode='valid'))
	model.add( BatchNormalization())


	model.add(Convolution2D( nClasses , 1, 1, border_mode='valid',))

	model.outputHeight = model.output_shape[-2]
	model.outputWidth = model.output_shape[-1]


	model.add(Reshape(( nClasses ,  model.output_shape[-2]*model.output_shape[-1]   ), input_shape=( nClasses , model.output_shape[-2], model.output_shape[-1]  )))
	
	model.add(Permute((2, 1)))
	model.add(Activation('softmax'))

	if not optimizer is None:
		model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )
	
	return model

