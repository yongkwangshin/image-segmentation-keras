import argparse
import glob
import os



parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )


args = parser.parse_args()


images_path = args.train_images
segs_path = args.train_annotations



img = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
img.sort()	
seg  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
seg.sort()
	
n=len(img)
m=len(seg)
print('n',n,'m',m)


print(img[0])
a=os.path.split(img[0])#[1]
print(a)
b=os.path.basename(img[0])
print(b)
a=os.path.splitext(img[0])
print(a)


for i in range(len(img)):
	for j in range(len(seg)):
		a=os.path.basename(seg[j])
		b=os.path.basename(img[i])
		# if a==b:
		# 	break
		# del seg[i]
			
			



# n=len(img)
# m=len(seg)
# print('n',n,'m',m)
# if img[0] == seg[0]:
# 	count+=1
# 	print ('dddddddddddddddddddddddddddddddddddd')





# n=3
# m=4
# for i in range(n):
# 	for j in range(m):
# 		if i==j:
# 			break
# 		print('i',i,'j',j)