import numpy as np
import cv2
import math

#------------Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#------------Eye Detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#------------Image to overlay
shades = cv2.imread('shades.png')


def warp(img):
	global shades
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	subimg = shades[65:145,25:250] #cropped image

	pL1=[] #top left
	pL2=[] #bottom left
	pR1=[] #top right
	pR2=[] #bottom right

	####### Detect face
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	X=0
	Y=0
	for (X,Y,w,h) in faces:
		cv2.rectangle(img,(X,Y),(X+w,Y+h),(255,0,0),2)
		roi_gray = gray[Y:Y+h, X:X+w]
		roi_color = img[Y:Y+h, X:X+w]
		
		eyes = eye_cascade.detectMultiScale(roi_gray)
		
		if(len(eyes) != 2):
			return img
		if(len(eyes) == 2):
			temp1 = eyes[0]
			temp2 = eyes[1]
			if(eyes[0][0]>eyes[1][0]):
				temp1 = eyes[1]
				temp2 = eyes[0]
		
			(exL,eyL,ewL,ehL)=temp1
			(exR,eyR,ewR,ehR)=temp2
		
			##### Centres of eye rectangle
			cenLx,cenLy = int(exL+ewL/2),int(eyL+ehL/2)
			cenRx,cenRy = int(exR+ewR/2),int(eyR+ehR/2)
			
			rot = math.atan2((cenRy-cenLy),(cenRx-cenLx))  # Rotation Angle
			
			# initialize corner points
			pL1 = (exL-cenLx),(eyL-cenLy)
			pL2 = (exL-cenLx),(eyL+ehL-cenLy)
			pR1 = (exR-cenRx+ewR),(eyR-cenRy)
			pR2 = (exR-cenRx+ewR),(eyR+ehR-cenRy)
			
			########## Points after rotation
			x = pL1[0]*math.cos(rot) - pL1[1]*math.sin(rot)
			y = pL1[1]*math.cos(rot) + pL1[0]*math.sin(rot)
			pL1 = x,y
		
			x = pL2[0]*math.cos(rot) - pL2[1]*math.sin(rot)
			y = pL2[1]*math.cos(rot) + pL2[0]*math.sin(rot)
			pL2 = x,y
		
			x = pR1[0]*math.cos(rot) - pR1[1]*math.sin(rot)
			y = pR1[1]*math.cos(rot) + pR1[0]*math.sin(rot)
			pR1 = x,y
		
			x = pR2[0]*math.cos(rot) - pR2[1]*math.sin(rot)
			y = pR2[1]*math.cos(rot) + pR2[0]*math.sin(rot)
			pR2 = x,y
		
			x = pL1[0] + cenLx
			y = pL1[1] + cenLy
			pL1 = [int(x),int(y)]
			x = pL2[0] + cenLx
			y = pL2[1] + cenLy
			pL2 = [int(x),int(y)]
		
			x = pR1[0] + cenRx
			y = pR1[1] + cenRy
			pR1 = [int(x),int(y)]
			x = pR2[0] + cenRx
			y = pR2[1] + cenRy
			pR2 = [int(x),int(y)]
		
			######## Perspective transform the Image
			pts1 = np.float32([[pR2[0]+X,pR2[1]+Y],[pL2[0]+X,pL2[1]+Y],[pL1[0]+X,pL1[1]+Y],[pR1[0]+X,pR1[1]+Y]])
			pts2 = np.float32([[subimg.shape[1],subimg.shape[0]],[0,subimg.shape[0]],[0,0],[subimg.shape[1],0]])
			
			M = cv2.getPerspectiveTransform(pts2,pts1)
			
			# Initialize final image
			img3 = np.zeros((img.shape[0],img.shape[1],3))
			for i in range(img.shape[0]):
				for j in range(img.shape[1]):
					img3[i][j]=(255,255,255)
			
			
			############ Warp image
			img3 = cv2.warpPerspective(subimg,M,(img.shape[1],img.shape[0]))

			########### Overlay Image
			for i in range(img.shape[0]):
				for j in range(img.shape[1]):
					if((img3.item(i,j,0)==255 and img3.item(i,j,1)==255 and img3.item(i,j,2)==255) or (img3.item(i,j,0)==0 and img3.item(i,j,1)==0 and img3.item(i,j,2)==0)):
						img3.itemset((i,j,0),img.item((i,j,0)))
						img3.itemset((i,j,1),img.item((i,j,1)))
						img3.itemset((i,j,2),img.item((i,j,2)))
			return img3

	return img

#----------Start capturing from webcam
cap = cv2.VideoCapture(0)


End_of_Video = False
while(1):
	
	ret, img = cap.read()
	height = img.shape[0]
	width = img.shape[1]
	
	# resize to suitable size
	img = cv2.resize(img,(int(0.5*width), int(0.5*height)), interpolation = cv2.INTER_CUBIC)
	if ret==False:
		End_of_Video = True
		break 
	
	warped_img = warp(img)
	cv2.imshow("Funk",warped_img)
	
	k = cv2.waitKey(1)
	if k == 27:
		break

cv2.destroyAllWindows()