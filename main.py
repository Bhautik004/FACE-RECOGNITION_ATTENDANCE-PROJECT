import numpy as np
import face_recognition
import cv2

imgElon = face_recognition.load_image_file('imageBasic/Elon_Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('imageBasic/elon_musk_test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)



faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceTest = face_recognition.face_locations(imgTest)[0]
encodeElonTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceTest[3],faceTest[0]),(faceTest[1],faceTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeElonTest)
faceDis = face_recognition.face_distance([encodeElon],encodeElonTest)
print(results)



cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Musk Test',imgTest)
cv2.waitKey(0)
