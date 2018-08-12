
import numpy as np
import cv2
import glob
import itertools


def getImageArr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):

	try:
		img = cv2.imread(path, 1) #경로에서 불러온다, 1은 컬러임을 의미

		if imgNorm == "sub_and_divide":
			img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1 # img를 (w,h) 사이즈로 바꿔준다
		elif imgNorm == "sub_mean": # imgNorm 어차피 sub_mean 이니 얘가 실행된다
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32) # 정수를 실수로 바꿔서 밑에 3줄 작동할 수 있게 만든다
			img[:,:,0] -= 103.939 # 왜 하필 103.939 랑 116.779 그리고 123.68 이냐면 이 코드를 짠 사람들 프로젝트에서 mean이 103,116,123 이기 때문이다. 우리는 우리 평균을 찾아내서 여기 부분 고치면 된다.
			img[:,:,1] -= 116.779
			img[:,:,2] -= 123.68
		elif imgNorm == "divide":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img = img/255.0

		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img
	except Exception, e:
		print path , e
		img = np.zeros((  height , width  , 3 ))
		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img
# 출력 : path(경로)에 있는 이미지파일을 imgNorm에 따라 처리를 거친 img가 출력된다




def getSegmentationArr( path , nClasses ,  width , height  ):

	seg_labels = np.zeros((  height , width  , nClasses )) # 세로,가로,클래스 의 행렬을 만든다.
	try:
		img = cv2.imread(path, 1) # path의미: 불러올 이미지경로, 1의미: 컬러로 불러오겠
		img = cv2.resize(img, ( width , height )) # 리사이즈한다.
		img = img[:, : , 0] # 이미지에서 블루채널만 추출한다. (거기에 무슨 클래스인지 정보가 담겨있다. 그리고 그 정보는 이 코드를 짠 사람들이 임의로 블루 채널에 넣어준거같다 - 추측)

		for c in range(nClasses): # 클래스 갯수만큼 for 돌린다
			seg_labels[: , : , c ] = (img == c ).astype(int) #  seg_labels에 클래스 갯수만큼 채널을 만든다. 그리고 해당 클래스값이면 1 아니면 0이 되도록 한다. 예를 들어 우리 프로젝트에서는 점자블록 클래스 하나만 존재하고 사진상에 점자블록이 있는 위치는 1, 나머지 부분은 0이 된다. 개인적인 생각인데 클래스가 점자블록과 점자블록 아닌것 두개 나올거 같기도 하다.
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	except Exception, e:
		print e
		
	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels
# 출력 : 컬러 이미지에서 블루픽셀만 남기고 나머지는 지운다.(행렬 자체를 삭제함) 그리고 블루픽셀이 클래스랑 같으면 1, 아니면 0을 넣는다. 1,0이 담긴 seg_lebels 행렬을 출력한다.


def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):
	
	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'

	images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )	# jpg, png, jpeg 확장자를 가진 이미지파일을 불러오고 images에 저장한다.
	images.sort()	 #정렬한다
	segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )	# jpg, png, jpeg 확장자를 가진 이미지파일을 불러오고 segmentations에 저장한다.
	segmentations.sort()	 #정렬한다

	assert len( images ) == len(segmentations) # images와 segmentations가 가진 파일 갯수가 같은지 확인한다.
	for im , seg in zip(images,segmentations):
		assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

	zipped = itertools.cycle( zip(images,segmentations) )
# itertools : https://kimdoky.github.io/python/2017/10/17/library-book-chap4-8.html    http://hamait.tistory.com/803  # zip : wikidocs.net/32#zip
	while True:
		X = []
		Y = []
		for _ in range( batch_size) :
			im , seg = zipped.next()
			X.append( getImageArr(im , input_width , input_height )  )
			Y.append( getSegmentationArr( seg , n_classes , output_width , output_height )  )

		yield np.array(X) , np.array(Y)


# import Models , LoadBatches
# G  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_train/" ,  "data/clothes_seg/prepped/annotations_prepped_train/" ,  1,  10 , 800 , 550 , 400 , 272   ) 
# G2  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_test/" ,  "data/clothes_seg/prepped/annotations_prepped_test/" ,  1,  10 , 800 , 550 , 400 , 272   ) 

# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( 800 , 550 )  )
# m.fit_generator( G , 512  , nb_epoch=10 )


