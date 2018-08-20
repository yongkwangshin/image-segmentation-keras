import os
file_path = os.path.dirname( os.path.abspath(__file__) )

print(__file__)
print(os.path.abspath(__file__))
print(file_path)
"""
output

test.py
/home/sky/gitshin/image-segmentation-keras/test.py
/home/sky/gitshin/image-segmentation-keras
"""

path="/home/sky/gitshin/image-segmentation-keras/test.h5"
# m=Model.(  img_input , output  )
# m.load_weights(path)
# https://keras.io/models/about-keras-models/#about-keras-models


def test( l=3):
	l=1
	print(l)
test()



from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D