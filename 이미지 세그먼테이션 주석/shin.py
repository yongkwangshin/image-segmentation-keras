import numpy as np
import cv2

"""
★주의할점
한글들어간 .py 파일은 prompt에서 실행 안됨, 한글 부분 지우고 실행하기
파이썬은 C언어랑 달리 중괄호{}를 사용 안한다.
띄워쓰기 한개만 되있으면 에러뜬다.
if랑 for 다음에 : 있어야한다. 만약 없으면 엔터쳤을때 Tab 안먹힌 상태로 줄바꿔진다.
kernel(커널) = 필터 = weights = convolution matrix = mask 이다
"""

컨트롤 +마우스올림 => 유용하다!!
"""
★비쥬얼스튜디오 코드 사용법 -> 단축키 인터넷에 많으니까 검색해보기
shift 엔터 : 해당줄을 터미널에서 실행한다
컨트롤 +마우스올림 : 마우스올린곳에 변수있으면 변수가 어디처음 쓰였는지나옴, 내장함수면 어떤함수인지 나옴
컨트롤 위아래 : 화면 올림 내림
컨트롤 좌우 : 의미단위 뛰어넘기
알트 위아래 : 행 끌어올리기 내리
컨트롤 F2 : 변수 한꺼번에 바꾸기
컨트롤 슬라이스 : 한꺼번에 주석처리/해
쉬프트 del : 행 삭제
"""


"""
★ prompt 사용법
conda info --envs : 콘다에 설치되있는 가상환경 목록보기
python -m pip install --upgrade pip : 최근에 뭔가 업데이트하라고 해서 기록해둠
"""


"""
★ 파이썬이 32bit 64bit 인지 확인하는 법
import platform
print(platform.architecture() )
"""


★밑에 테스트 코드들은 c:\test 에서 실행하면됨
그리고 아래에 있는 테스트 코드들 따로 실행할때 import numpy cv2 해줘야 실행된다.



# 그림판에서 빨간 초록 파랑으로 칠한 그림 불러와서 프린트해보기
# # RGB tset
# R=cv2.imread('red.jpg',1)
# G=cv2.imread('green.jpg',1)
# B=cv2.imread('bule.jpg',1)
# # cv2.imshow('load',img)
# # cv2.waitKey()
# print('red')
# print(R)
# print('green')
# print(G)
# print('blue')
# print(B)
# a=[[[0,0,0],[0,0,0]]]
# a=np.array(a)
# cv2.imwrite( 'a_test.jpg',a )


# 이미지 불러와서 열어보고 다른이름으로 저장해보기
# # open image & save
# img = cv2.imread('abc.jpg', 0 )
# cv2.imshow( 'load image', img)
# print(img)
# key = cv2.waitKey(0) 
# cv2.imwrite( 'abc_222.jpg', img )


# 리스트 만들고 a[0]으로 원하는 부분만 출력해보기
# # make list & access
# a=[1, 2]
# print(a[0])


# 배열 만들고 배열을 이미지로 저장해보기
# # make list & access as image file
# a=[[254,1],[2, 255]]
# print(a)
# a=np.array(a) # make numpy array !! must have to !!
# cv2.imwrite( 'abc_222.jpg', a ) # save


# # save color that i want
# # [[36,27,237],[77,177,35],[204,71,63]]
# # [[36,27,237],[36,27,237],[36,27,237]]
# # [[77,177,35],[77,177,35],[77,177,35]]
# # [[204,71,63],[204,71,63],[204,71,63]]
# # [[255,255,255],[255,255,255],[255,255,255]]
# a=[  [[36,27,237],[77,177,35],[204,71,63]]  ,  [[0,0,0],[0,0,0],[0,0,0]]  ,  [[255,255,255],[255,255,255],[255,255,255]]   ]
# # a=[0 ,0, 0, 0, 0, 0]
# # print(a)
# a=np.array(a)
# cv2.imwrite( 'pcha.png', a )
# #load
# img=cv2.imread('pcha.png',1)
# print(img)


# 배열 만들고 출력해보기
# # RGB representaion in 2x2 matrix
# a=[  [[1,2,3],[4,5,6]]  , [[7,8,9],[10,11,12]]  ]
# # [[1,2,3],[4,5,6]] is first col
# a=np.array(a)
# print('\na') # \n : new line
# print(a)
# print('\na[0]')
# print(a[0])
# print('\na[1]')
# print(a[1])
# print('\n\na[0][0]')
# print(a[0][0])
# print('\na[0][1]')
# print(a[0][1])
# col=len(a) #col
# row=len(a[0]) #row
# print('col=',col,'row=',row)
# for i in range(col):
#    for j in range(row):
#         print('i=',i,'j=',j)
#         print(a[i][j])


# 불대수를 파이썬 행렬에 적용해보기
# # test logic in matrix
# a=[  [[1,2,3],[4,5,6]]  , [[7,8,9],[10,11,12]]  ]
# a=np.array(a)
# print(a[0][0]!=[1,1,1])
# print(sum(a[0][0]==[255,255,255]))
# print(a[0][0]>[100,100,100])


# 이미지 불러와서 행, 열 측정해보기
# # load image & measure size
# a = cv2.imread('pingtest_small.png', 1 )
# col=len(a) #col
# row=len(a[0]) #row
# print("col=",col,'row=',row)



# 불대수 적용해서 이미지 처리하기
# # logic apply to image matirx in regular sequence
# a = cv2.imread('pingtest_small.png', 1 )
# print(a)
# col=len(a) #col
# row=len(a[0]) #row
# print("col=",col,'row=',row)
# for i in range(col):
#    for j in range(row):
#         # if sum( a[i][j] != [255,255,255] )==3:
#         if sum(a[i][j]<[240,240,240])==3:
#         # if sum(a[i][j]<[240,240,240])>2.5:
#            a[i][j]=[0,0,0] 
#         else:
#             a[i][j]=[255,255,255] 
# cv2.imwrite('pring.png',a)


# astype로 저장 형태 바꾸
# # np.float32: int 2 -> float 2.0
# a=[  [[1,2,3],[4,5,6]]  , [[7,8,9],[10,11,12]]  ]
# print('list')
# print(a)
# a=np.array(a)
# print('np array')
# print(a)
# a = a.astype(np.float32)
# print('np float32')
# print(a)


# 파이썬 행렬 모양에 대한 연
# # shape of matrix turm
# a=np.zeros((5,3,2,4))
# print('---a')
# print(a)
# print('---a[0]')
# print(a[0])
# print('---b')
# b=np.ones((3,2,4))
# print(b)
# print('therefore a[0]=b')
# # np.zeros((5,4,2,3)=>shape turm!! ) or np.ones


# 2차원 행렬에서 axis 바꿔보기
# study about axis in 2D matix
print("""2D matix axis test""")
a=[ [1,2,3,4] , [5,6,7,8] , [9,10,11,12] , [13,14,15,16]  ]
a=np.array(a)
#print(a)
print('')
out=np.max(a,axis=0)
print('axis=0')
print(out)
print('')
out2=np.max(a,axis=1)
print('axis=1')
print(out2)


# 3차원 행렬에서 axis 바꿔보기
# study about axis in 3D matix
#2 2 4 matrix
print("""3D matix axis test""")
b=[ [ [1,2,3,4] , [5,6,7,8] ] , [ [9,10,11,12] , [13,14,15,16] ] ]
b=np.array(b)
print('b')
print(b)
print('shape of b')
print(np.shape(b))
print('')
out=np.max(b,axis=0)
print('axis=0')
print(out)
print('')
out2=np.max(b,axis=1)
print('axis=1')
print(out2)
print('')
out3=np.max(b,axis=2)
print('axis=2')
print(out3)


# study about axis in 3D matix
# 3 2 4 matix
c=[ [ [1,2,3,4] , [5,6,7,8] ] , [ [9,10,11,12] , [13,14,15,16] ] , [ [17,18,19,20] , [21,22,23,24] ]  ]
c=np.array(c)
print('c')
print(c)
print('shape of c')
print(np.shape(c))
print('')
out=np.max(c,axis=0)
print('axis=0')
print(out)
print('')
out2=np.max(c,axis=1)
print('axis=1')
print(out2)
print('')
out3=np.max(c,axis=2)
print('axis=2')
print(out3)


# 슬라이스 이용해서 행렬에 접근해보기
# slice the list
c=[ [ [1,2,3,4] , [5,6,7,8] ] , [ [9,10,11,12] , [13,14,15,16] ] , [ [17,18,19,20] , [21,22,23,24] ]  ]
c=np.array(c)
print('c')
print(c)
print('c[0]')
print(c[0])
print('c[:,:,0]')
print(c[:,:,0])
print('c[:,0,0]')
print(c[:,0,0])
print('c[0,0,0]')
print(c[0,0,0])
print('c[0,:,:]')
print(c[0,:,:])
print('c[0,0,:]')
print(c[0,0,:])


# reshape func
before = np.arange(24)
print('before')
print(h)
after=before.reshape(2,3,4)
print('after')
print(h)


# i[0]이 뭔지 뽑아내보기
# first data 
i=[ [ [ [1,2,3] , [4,5,6] ] , [  [7,8,9] , [10,11,12] ] ] , [ [ [13,14,15] , [16,17,18] ] , [  [19,20,21] , [22,23,24] ] ]   ]
i=np.array(i)
print('i')
print(i)
print('np.shape(i):')
print(np.shape(i))
print('i[0]')
print(i[0])



# 텐서플로우로 덧셈구현
import tensorflow as tf
a=tf.constant(5)
b=tf.constant(2)
c=tf.add(a,b)
print(c) # it isn't work that i think.
sess=tf.Session()
out=sess.run(c)
print(out) # this is what i want
# => 텐서플로우는 항상 Session으로 run 시켜줘야 하는거 같음
