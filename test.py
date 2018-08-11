import argparse
import glob



parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )



train_images_path = args.train_images
train_segs_path = args.train_annotations



img = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
img.sort()	
seg  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
seg.sort()
	

for i in range(len(img))
	for j in range(len(seg))
		print('i',i,'j',j)
