# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:30:48 2022

@author: User
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
#import time

#Création de l'image alpha à partir du chemin de l'image et d'un seuil
def alpha(img, seuil):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, seuil, 255, cv2.THRESH_BINARY_INV)
    img_alpha = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    return img_alpha

#Affichage des images (pour les tests)     
def affichage(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show() 

#Affichage d'une image en modifiant le fond détecté grâce à son alpha en noir
def test_incrustation(img,img_alpha,seuil):
    img2 = img.copy()
    img_alpha = cv2.cvtColor(img_alpha, cv2.COLOR_BGR2GRAY)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_alpha[i,j] < seuil:
                img2[i,j] = (0,0,0)
    affichage(img2)
  
#Application d'une dilatation puis d'une érosion
def fermeture(nb_neighbours,iteration,img):
    kernel = np.ones((nb_neighbours,nb_neighbours), np.uint8)
    dilatation = cv2.dilate(img,kernel, iterations=iteration)
    dilatation = cv2.cvtColor(dilatation, cv2.COLOR_BGR2RGB)
    erosion = cv2.erode(dilatation,kernel, iterations=iteration)
    return erosion

#Filtre sepia
def filter_sepia(img):
    kernel = np.float32([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
    return cv2.transform(img, kernel)

#Ajout d'une image sur une autre
def incrustation(frame,img,img_alpha,object_detected,n_object):
    left_top = [max(0,object_detected[n_object,0]),max(object_detected[n_object,1],0)]
    right_bottom = [min(frame.shape[1],object_detected[n_object,0]+object_detected[n_object,2]),min(frame.shape[0],object_detected[n_object,1]+object_detected[n_object,3])]
    diff_width = right_bottom[0]-left_top[0]
    diff_height = right_bottom[1]-left_top[1]


    # Read the images
    foreground = cv2.resize(img,(diff_width,diff_height))
    background = frame[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    alpha_img = cv2.resize(img_alpha,(diff_width,diff_height))
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha_img = alpha_img.astype(float)/255  
    """print(str(alpha_img.shape[0])+","+str(alpha_img.shape[1])+","+str(alpha_img.shape[2])+"/"+str(foreground.shape[0])+","+str(foreground.shape[1])+","+str(foreground.shape[2])+"/"+str(background.shape[0])+","+str(background.shape[1])+","+str(background.shape[2]))"""
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha_img, foreground)    
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha_img, background)
    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    frame[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]] = outImage

#Création de rectangle blanc ou noir à partir de coordonnées
def rectangle(image,start_point,end_point,enable):
    # Line thickness of -1 px
    # Thickness of -1 will fill the entire shape
    black = (0,0,0)
    white = (255,255,255)
    thickness = -1
    if enable == 1:
        cv2.rectangle(image, start_point, end_point, white, thickness)
    else:
        cv2.rectangle(image, start_point, end_point, black, thickness)


#Modification des variables (on/off) après un clic sur des coordonnées précises
def on_click(event, x, y, p1, p2):

        
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10<y<40:    
            if 10<x<40:
                enable["sepia"] *= -1
            elif 50<x<80:
                enable["hat"] *= -1
            elif 90<x<120:
                enable["eyes"] *= -1
            elif 130<x<160:
                enable["nose"] *= -1
            elif 170<x<210:
                enable["balls"] *= -1
            elif 210<x<240:
                enable["circus"] *= -1
                
                
# Load the cascades
face_cascade = cv2.CascadeClassifier('CascadeClassifier/haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier('CascadeClassifier/haarcascade_eye_tree_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier('CascadeClassifier/haarcascade_mcs_nose.xml')

#Load the pictures
clown_hat = cv2.imread('chapeau.png')
clown_eye = cv2.imread('oeil.jpg')
clown_nose = cv2.imread('nez.jpg')
ball = cv2.imread('ball.jpg')
circus_background = cv2.imread('background_circus.jpg')
    
#Define position of buttons
button_sepia = np.array([[10,10,30,30]])
button_hat = np.array([[50,10,30,30]])
button_eye = np.array([[90,10,30,30]])
button_nose = np.array([[130,10,30,30]])
button_balls = np.array([[170,10,30,30]])
button_circus = np.array([[210,10,30,30]])

#Define position of frame
frame_size = np.array([[0,0,640,480]])


#Binarizations
alpha_hat = alpha(clown_hat, 210)
#alpha_hat = fermeture(3, 20, alpha_hat)
#affichage(alpha_hat)
alpha_eye = alpha(clown_eye, 250)
#affichage(alpha_eye)
alpha_nose = alpha(clown_nose, 210)
#affichage(alpha_nose)
alpha_ball = alpha(ball,200)
alpha_ball = fermeture(3,10,alpha_ball)
#affichage(alpha_ball)
alpha_circus = alpha(circus_background, 100)
#affichage(alpha_circus)




#Creation of table
balls_position = []
n_balle = 0
#i correspondent aux balles
for i in range(30,630,60):
    liste = []
    #j correspondent aux temps
    #for j in range(0,440,1): --> plus lent
    for j in range(0,440,10):
        #Pour chaque balle à un moment donné on donne des coordonnées, le modulo permet de réinitialiser les balles en haut une fois qu'elles ont atteint le bas de l'image
        liste.append({"x":i, "y":(j+44*n_balle)%440})
    balls_position.append(liste)
    n_balle +=1
    




#Dictionnary of enables
enable={}
enable["sepia"] = -1
enable["hat"] = -1
enable["eyes"] = -1
enable["nose"] = -1
enable["balls"] = -1
enable["circus"] = -1

#Launch the webcam
videoWebcam = cv2.VideoCapture(0)
#Capture video
vid_cod = cv2.VideoWriter_fourcc(*"XVID")
output = cv2.VideoWriter("RAMANITRA_MONTALAND_DEMO.mp4", vid_cod, 4.0, (640,480))

temps = 0
while True:
    valeurRetour, frame = videoWebcam.read()
    
    cv2.namedWindow('Image de la webcam')
    cv2.setMouseCallback('Image de la webcam', on_click)
        
    
    if enable["sepia"] == 1:
        frame = filter_sepia(frame)
    
    if enable["circus"] == 1:
        alpha_frame = alpha(frame,95)
        
        incrustation(frame, circus_background, 255 - alpha_frame, frame_size,0)
        
    # Detect faces
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    eyes = eyes_cascade.detectMultiScale(frame, 1.1, 4)
    noses = nose_cascade.detectMultiScale(frame, 1.1, 4)
    
    
    if enable["hat"] == 1 : 
        for i in range(len(faces)):
            #Décalage de la tête pour que le chapeau soit au dessus de la tête plutôt que sur
            top_faces = faces.copy()
            top_faces[:,0] = faces[:,0] - faces[:,2]*0.60
            top_faces[:,1] = faces[:,1] - faces[:,3] 
            top_faces[:,2] = faces[:,2] * 2
            top_faces[:,3] = faces[:,1] - top_faces[:,1]
            incrustation(frame, clown_hat, alpha_hat, top_faces, i) 
            
    if enable["eyes"] == 1:
        for i in range(len(eyes)):
            for j in range(len(faces)):
                #Si l'oeil détecté se trouve dans un visage détecté
                if faces[j][0]<eyes[i][0]<faces[j][0]+faces[j][2] and faces[j][1]<eyes[i][1]<faces[j][1]+faces[j][3]:
                    incrustation(frame, clown_eye, alpha_eye, eyes, i)
                    break

            
    if enable["nose"] == 1:        
        for i in range(len(noses)):
            #Redimensionnage des nez pour qu'il soient plus rond
            round_noses = noses.copy()
            round_noses[:,0] = 0.5*(noses[:,2]-noses[:,3])
            round_noses[:,2] = noses[:,3]
            for j in range(len(faces)):
                if faces[j][0]<noses[i][0]<faces[j][0]+faces[j][2] and faces[j][1]<noses[i][1]<faces[j][1]+faces[j][3]:
                        incrustation(frame, clown_nose, alpha_nose, noses, i)
                        break
    
    if enable["balls"] == 1:
        #incrémentation de la variable qui permet de passer d'une ligne à l'autre et donc de modifier les coordonnées des balles
        temps += 1
        reste = temps % len(balls_position[0])
        
        #Incrustation de chaque balle
        for balle in range(len(balls_position)):
            foreground = cv2.resize(ball,(30,30))
            background = frame[balls_position[balle][reste]["y"]:30+balls_position[balle][reste]["y"],balls_position[balle][reste]["x"]:30+balls_position[balle][reste]["x"]]
            alpha_img = cv2.resize(alpha_ball,(30,30))
            # Convert uint8 to float
            foreground = foreground.astype(float)
            background = background.astype(float)
            # Normalize the alpha mask to keep intensity between 0 and 1
            alpha_img = alpha_img.astype(float)/255  
            # Multiply the foreground with the alpha matte
            foreground = cv2.multiply(alpha_img, foreground)    
            # Multiply the background with ( 1 - alpha )
            background = cv2.multiply(1.0 - alpha_img, background)
            # Add the masked foreground and background.
            outImage = cv2.add(foreground, background)
            frame[balls_position[balle][reste]["y"]:30+balls_position[balle][reste]["y"],balls_position[balle][reste]["x"]:30+balls_position[balle][reste]["x"]] = outImage
    
            

    # Gestion bouton
    rectangle(frame,(10,10),(40,40),enable["sepia"])
    rectangle(frame,(50,10),(80,40),enable["hat"])
    rectangle(frame,(90,10),(120,40),enable["eyes"])
    rectangle(frame,(130,10),(160,40),enable["nose"])
    rectangle(frame,(170,10),(200,40),enable["balls"])
    rectangle(frame,(210,10),(240,40),enable["circus"])

    
    #Incrustation des images sur les carrés
    incrustation(frame, clown_hat, alpha_hat, button_hat, 0)
    incrustation(frame, clown_eye, alpha_eye, button_eye, 0)
    incrustation(frame, clown_nose, alpha_nose, button_nose, 0)
    incrustation(frame, ball, alpha_ball, button_balls, 0)
    incrustation(frame, circus_background, alpha_circus, button_circus, 0)
    
    #Write video
    output.write(frame)
    #Show video
    cv2.imshow('Image de la webcam', frame) 
    
    
    #Press q to stop the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

videoWebcam.release()
output.release()
cv2.destroyAllWindows() 



