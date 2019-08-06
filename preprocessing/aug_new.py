# reads from the train.txt

# Data Augmentation : To increase the training examples and
# avoid overfitting we use various data augmentation techniques like
# horizontal flip,Gaussian blur followed by sharpening and image
# darkening.The training set is increased five folds to 15000 after
# applying the augmentation.


from imgaug import augmenters as iaa
import os
import cv2
import matplotlib.pyplot as plt
import random

seq1 = iaa.Sequential([
    iaa.Fliplr(1), # horizontally flip 50% of the images
])
seq2 = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0,1.5)), # blur images with a sigma of 0 to 3.0
	iaa.Sharpen(1,2) # blur images with a sigma of 0 to 3.0
])
seq3 = iaa.Sequential([
    iaa.Multiply(0.4) # blur images with a sigma of 0 to 3.0
])

seq4 = iaa.Sequential([
	iaa.Affine(shear = 30),
    iaa.AdditiveGaussianNoise(scale=0.01*255) # blur images with a sigma of 0 to 3.0
])

seq5 = iaa.Sequential([
    iaa.Affine(shear = 25), # blur images with a sigma of 0 to 3.0
    iaa.Dropout(p=0.02)
])


with open("total.txt") as f:
    content = f.readlines()
# print (content)


path = '/home/ayush/lane/ext_image/' 


train_file = open("train_aug.txt","w") 
val_file = open("val_aug.txt","w") 

train_file.write("image,labels\n")
val_file.write("image,labels\n")

sol=[]
dot=[]
fal=[]

for n in content[1:]:

	name = n.split(",")[0]
	lane_type = n.split(",")[1]

	print (name,lane_type)
	
	if int(lane_type) == 0 :
		sol.append(name)
	elif int(lane_type) == 1:
		dot.append(name)
	elif int(lane_type) == 2:
		fal.append(name)

	# plt.figure()
	img = cv2.imread(path+"/"+str(name),-1)
	cv2.imwrite("./ext_image_dual_aug/"+str(name)[:-4]+".jpg", img)

	# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
	# plt.show()
	for s,seq in zip([seq1,seq2,seq3,seq4,seq5],["seq1","seq2","seq3","seq4","seq5"]):
		
		images_aug = s.augment_images([img])

		if int(lane_type)==1 and  (seq == "seq4" or seq == "seq5"):
			
			cv2.imwrite("./ext_image_dual_aug/"+str(name)[:-4]+seq+".jpg", images_aug[0])
			
			if int(lane_type) == 0 :
				sol.append(str(name)[:-4]+seq+".jpg")
			elif int(lane_type) == 1:
				dot.append(str(name)[:-4]+seq+".jpg")
			elif int(lane_type) == 2:
				fal.append(str(name)[:-4]+seq+".jpg")

		elif int(lane_type)==2 and  (seq in ["seq4","seq5"]) :
			cv2.imwrite("./ext_image_dual_aug/"+str(name)[:-4]+seq+".jpg", images_aug[0])
			# train_file.write("%s,%d\n" %(str(name)[:-4]+seq+".jpg",int(lane_type))) 

			# if int(lane_type) == 0 or int(lane_type) == 1 :
			if int(lane_type) == 0 :
				sol.append(str(name)[:-4]+seq+".jpg")
			elif int(lane_type) == 1:
				dot.append(str(name)[:-4]+seq+".jpg")
			elif int(lane_type) == 2:
				fal.append(str(name)[:-4]+seq+".jpg")

		elif seq in ["seq1","seq2","seq3"]:
			cv2.imwrite("./ext_image_dual_aug/"+str(name)[:-4]+seq+".jpg", images_aug[0])
			# train_file.write("%s,%d\n" %(str(name)[:-4]+seq+".jpg",int(lane_type))) 

			# if int(lane_type) == 0 or int(lane_type) == 1 :
			if int(lane_type) == 0 :
				sol.append(str(name)[:-4]+seq+".jpg")
			elif int(lane_type) == 1:
				dot.append(str(name)[:-4]+seq+".jpg")
			elif int(lane_type) == 2:
				fal.append(str(name)[:-4]+seq+".jpg")

random.shuffle(sol)
random.shuffle(dot)
random.shuffle(fal)

print len(sol)
print len(dot)
print len(fal)

train_set_sol = sol[:int(0.8*len(sol))] 
val_set_sol = sol[int(0.8*len(sol)):]
train_set_dot = dot[:int(0.8*len(dot))] 
val_set_dot = dot[int(0.8*len(dot)):] 
train_set_fal = fal[:int(0.8*len(fal))] 
val_set_fal = fal[int(0.8*len(fal)):]

print len(val_set_sol)
print len(val_set_dot)
print len(val_set_fal)

for i in train_set_sol:
	train_file.write("%s,%d\n" %(i,0)) 
for i in val_set_sol:
	val_file.write("%s,%d\n" %(i,0)) 
for i in train_set_dot:
	train_file.write("%s,%d\n" %(i,1)) 
for i in val_set_dot:
	val_file.write("%s,%d\n" %(i,1)) 
for i in train_set_fal:
	train_file.write("%s,%d\n" %(i,2)) 
for i in val_set_fal:
	val_file.write("%s,%d\n" %(i,2)) 



train_file.close()
val_file.close()
		
