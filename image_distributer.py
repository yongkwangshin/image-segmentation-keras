import argparse
import glob
import os
" 0 how to execute "
"""
python  image_distributer.py --train_images="data1/" --train_annotations="data2/"

train_images is directory to be changed
train_annotations is directory to be compare
"""


" 1 set argparse "
parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
args = parser.parse_args()


" 2 get path "
images_path = args.train_images
segs_path = args.train_annotations
img = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
img.sort()	
seg  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
seg.sort()


" 3 comparing img,seg and then get same position index  "
img_name=[]
seg_name=[]
count=[]

n=len(img)
m=len(seg)
for i in range(n):
    for j in range(m):
        img_path=img[i]
        seg_path=seg[j]
        img_name=os.path.splitext(os.path.basename(img_path))[0]
        seg_name=os.path.splitext(os.path.basename(seg_path))[0]
        if img_name == seg_name:
            count.append(j)
            break


" 4 making new seg2 by index "
seg2=[]
for j in count:
    seg2.append(seg[j])


" 5 move directory path and move picture "
pic_path='data2/'
os.chdir(pic_path)

new_path='data3/'
for i in range(len(seg2)):
    seg2_path=seg2[i]
    seg2_pre_name=os.path.basename(seg2_path) # include .jpg .png .jpeg
    seg2_name=os.path.splitext( seg2_pre_name )[0]
    new_path_and_name = new_path + seg2_pre_name
    os.renames( seg2_pre_name, new_path_and_name)


"""
0,1,2
외부로부터 변수를 입력받아 내부에서 바로 쓴다
argparse.ArgumentParser.add_argument("--변수이름",type=str)
간편히 만들변수=argparse.ArgumentParser.parse_args.변수이름

2
경로 = "data/" -> py 실행되는 폴더 안에 data 폴더
glob.glob( 경로 + 확장자 )


3,5
os.path.basename("경로/파일.확장자") ->출력: 파일.확장자
img_name=os.path.splitext( os.path.basename("경로/파일.확장자") )[0] ->출력: 파일 (확장자 제외)

4,5
리스트.append(변수) -> 리스트 끝에 변수를 추가한다

0 실행명령
python  test.py --train_images="data1/" --train_annotations="data2/"

이름만 추출하기
path = "a/b/c/abc.txt"
result = os.path.splitext(os.path.basename(path))[0]
print(result)

탭 에러 해결책
TabError: inconsistent use of tabs and spaces in indentation
-> use only tab or space


"""