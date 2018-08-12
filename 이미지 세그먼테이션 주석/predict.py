import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser() #parser이라는 객체를 만든다.
parser.add_argument("--save_weights_path", type = str  ) 
parser.add_argument("--epoch_number", type = int, default = 5 )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--n_classes", type=int ) # "--n_classes ~" 얘의 값을 parser.parse_args.n_classes라는 항목으로 만든다.

args = parser.parse_args() # parser.parse_args.n_classes 를 짧게 args.n_classes 쓰려고 객체 args 만듬

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ] # 위에 모델 중에 원하는 모델을 modelFN에 넣는다, 상황에 따라 원하는 모델로 바꿀 수 있으므로 편리하다. (python 자료형인 딕셔너리 알아야 위에 코드 이해가능)

m = modelFN( n_classes , input_height=input_height, input_width=input_width   ) # 모델을 m에 장착해준다. h,w는 입력 기본값이 이미 함수안에 있어서 입력 안해도됨 (Models 폴더에서 하나 들어가서 데이터 입력형태 보면 이해가능)
m.load_weights(  args.save_weights_path + "." + str(  epoch_number )  ) # 예시: str("life is") 출력 -> 'life is'
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy']) #  m.compile = keras.models.Sequential.compile 임 (Segnet 보면 이렇게 나옴)
# 윗줄의 의미 : lossfunc은 loss를, optimizer는 optimizer를 정확도 측정 파라미터는 metrics를 사용한다.
# https://keras.io/models/sequential/ 케라스 공식 문서,가이드
output_height = m.outputHeight # outputHeight 못찾음 ㅠㅠ ★★★★★★★★★★★★★★★★★★★
output_width = m.outputWidth
# glob : path에 있는 "*.jpg" 랑 "*.png" 랑 "*.jpeg"를 image를 images에 쌓는다. "*.jpg"의 의미 : 확장자가 jpg인 파일 모두 
images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort() # sort : images를 순서대로 정렬하는 함수

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]
# class 갯수만큼 랜덤으로 1행3렬 행렬을 만들어서(이게 RGB 역할을 한다) 리스트 colors에 넣는다
for imgName in images: # 정렬한 이미지를 순서대로 imgName에 넣는다.
	outName = imgName.replace( images_path ,  args.output_path ) # images_path이랑 args.output_path는 13,23번째 줄에 있음
	X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height  ) #이미지와 앞에 저장한(24,25번째 줄) 입력 w,h를 getImageArr함수에 넣어서 출력을 얻는다
	pr = m.predict( np.array([X]) )[0] # 왜 첫번째 행만 가져다 쓰는지 모르겠다, 입력이미지의 형태를 알아야할듯 ★★★★★★★★★★★★★★★★★★★
	pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 ) # argmax: 값이 최대인 부분의 인덱스를 찾아냄 axis=2 : 어디서 최대값인지 정해주는 인자이다. 행에서 최대인지 열에서 최대인지를 정해준다.
	seg_img = np.zeros( ( output_height , output_width , 3  ) ) # 0으로 만든 행렬을 seg에 저장한다
	for c in range(n_classes):# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
		seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8') # astype : unsign 8비트 int
		seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8') # pr[:,:]==c 는 행렬 대 정수 비교지만 파이썬에서는 행렬의 각각의 원소들과 c가 비교된다. 예를 들어 [ [1,2,3] , [4,5,6] ] ==1 하면 [ [Ture,F,F] , [F,F,F]  ] 나온다. 이런 특징을 따로 지칭하는 이름있는데 지금 생각 안남ㅠ
		seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8') # 아무튼 여기서 seg_img의 채널에서 클래스에 해당하는 값을이랑 비교 해서 그 클래스가 맞으면 colors(색깔)을 부여해준다. 그리고 그 색깔을 seg_img에 더해준다. 더해줄때 seg_img는 영행렬 이므로 사실상 저장해주는것임.
	seg_img = cv2.resize(seg_img  , (input_width , input_height )) # 리사이즈
	cv2.imwrite(  outName , seg_img ) # predict.py 가 있는 폴더에 저장한다.

