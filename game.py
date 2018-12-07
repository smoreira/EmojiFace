# Developed by Sandro Silva Moreira - moreira.sandro@gmail.com
# for demonstrate how apply techniques of Deep Learning 

#In this project a Neural Network has trained with 7 expressions corresponding a 7 Emojis... 
#to increase the points the user must imitate the emoji

import os
import numpy as np
import cv2
import random
from keras.preprocessing import image

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)

#Decrease frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);

#-----------------------------
#Neural Model - face expression

from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

#-----------------------------

emotions = ('Raiva', 'Desgosto', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro')

sequencia = []
pontos = 0

def remove_item(my_list,*args):
    deletar = list(args)
    for item in deletar:
        while item in my_list:
            my_list.remove(item)
    return my_list

for _ in range(5000):
     sequencia.append(random.randint(0,6))


remove_item(sequencia,1) #i removed the expression disgust because is very difficult 


while(True):
	ret, img = cap.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
		predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		#max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(predictions[0])
		
		emotion = emotions[max_index]
		
		#write emotion text above rectangle
		cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		
		#-------------------------

		if max_index == sequencia[0]:
			#os.system('afplay /System/Library/Sounds/Sosumi.aiff') #alert sound  - increase points
			del sequencia[0]
			pontos = pontos + 1

		#points on screen
		pontuacao = str(pontos) + " pontos"
		cv2.putText(img, pontuacao, (int(730), int(240)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

		#print faces on screen
		img2 = cv2.imread('./emojis/'+str(sequencia[0])+'.png',-1)
		img2 = cv2.resize(img2,(150,150))
		y_offset=50 
		x_offset=750

		y1, y2 = y_offset, y_offset + img2.shape[0] 
		x1, x2 = x_offset, x_offset + img2.shape[1] 
		alpha_s = img2[:, :, 3] / 255.0 
		alpha_l = 1.0 - alpha_s 
		
		for c in range(0, 3): 
			img[y1:y2, x1:x2, c] = (alpha_s * img2[:, :, c] + alpha_l * img[y1:y2, x1:x2, c]) 


	#------------------------------
	#logo
	logo = cv2.imread('logo.png',-1)
	ly_offset=450 
	lx_offset=50

	ly1, ly2 = ly_offset, ly_offset + logo.shape[0] 
	lx1, lx2 = lx_offset, lx_offset + logo.shape[1] 
	lalpha_s = logo[:, :, 3] / 255.0 
	lalpha_l = 1.0 - lalpha_s 

	for c in range(0, 3): 
		img[ly1:ly2, lx1:lx2, c] = (lalpha_s * logo[:, :, c] + lalpha_l * img[ly1:ly2, lx1:lx2, c]) 

	#----------------------------

	cv2.imshow('Emoji Face Game',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()