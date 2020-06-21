########## Importing Packages ############
import numpy as np
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
root=Tk()
root.geometry('500x570')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Augmented Reality Cam')
frame.config(background='light blue')
label = Label(frame, text="OpenAR",bg='light blue',font=('Times 32 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="demo2.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)



def hel():
   help(cv2)

def Contri():
   tkinter.messagebox.showinfo("Contributors","\n1. Parijat Dhar\n2. Nikunj Goyal \n")


def anotherWin():
   tkinter.messagebox.showinfo("About",'Augmented Reality Cam - OpenAR v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3')
                                    
   
########### Menu Section ###############
menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Open CV Docs",command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Augmented Reality Cam",command=anotherWin)
subm2.add_command(label="Contributors",command=Contri)

subm3 = Menu(menu)
menu.add_cascade(label='Press q to exit!')


############# For Exit #############
def exitt():
   exit()

########## For Opening Web Cam ###########  
def web():
   capture =cv2.VideoCapture(0)
   while True:
      ret,frame=capture.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
   capture.release()
   cv2.destroyAllWindows()

########## For Recording video using Web Cam ###########
def webrec():
   capture =cv2.VideoCapture(0)
   fourcc=cv2.VideoWriter_fourcc(*'XVID') 
   op=cv2.VideoWriter('RecordVid.avi',fourcc,11.0,(640,480))
   while True:
      ret,frame=capture.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('frame',frame)
      op.write(frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
   op.release()
   capture.release()
   cv2.destroyAllWindows()   

def webdet():
   ####### Importing all the media files ############
   cap = cv2.VideoCapture(0)
   imgTarget = cv2.imread('TargetImage.jpg')
   myVideo = cv2.VideoCapture('video.mp4')

   ####### Augmenting the video #########
   detection=False
   frameCounter = 0

   ########### Grabbing the First Frame ############
   success, myVid = myVideo.read()
   hT,wT,cT = imgTarget.shape
   myVid = cv2.resize(myVid, (wT,hT)) #for initial purpose, we will continue imposing an image as mask

   ########### ORB Detector ##############
   orb = cv2.ORB_create(nfeatures=1000)
   kp1, des1 = orb.detectAndCompute(imgTarget,None)
   # imgTarget = cv2.drawKeypoints(imgTarget,kp1,None)


   ########## Stacking Function for stacking the image ############
   def stackImages(imgArray,scale,lables=[]):
      sizeW= imgArray[0][0].shape[1]
      sizeH = imgArray[0][0].shape[0]
      rows = len(imgArray)
      cols = len(imgArray[0])
      rowsAvailable = isinstance(imgArray[0], list)
      if rowsAvailable:
         for x in range ( 0, rows):
               for y in range(0, cols):
                  imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW,sizeH), None, scale, scale)
                  if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
         imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
         hor = [imageBlank]*rows
         hor_con = [imageBlank]*rows
         for x in range(0, rows):
               hor[x] = np.hstack(imgArray[x])
               hor_con[x] = np.concatenate(imgArray[x])
         ver = np.vstack(hor)
         ver_con = np.concatenate(hor)
      else:
         for x in range(0, rows):
               imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
               if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
         hor= np.hstack(imgArray)
         hor_con= np.concatenate(imgArray)
         ver = hor
      if len(lables) != 0:
         eachImgWidth= int(ver.shape[1] / cols)
         eachImgHeight = int(ver.shape[0] / rows)
         print(eachImgHeight)
         for d in range(0, rows):
               for c in range (0,cols):
                  cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                  cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
      return ver


   ############# To find keyPoints and descriptors from the Target and Webcam image ##############
   while True:
      success, imgWebcam = cap.read()
      kp2, des2 = orb.detectAndCompute(imgWebcam, None)
      # imgWebcam = cv2.drawKeypoints(imgWebcam,kp2,None)

      if detection == False:
         myVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
         frameCounter=0
      else:
         if frameCounter == myVideo.get(cv2.CAP_PROP_FRAME_COUNT):
               myVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
               frameCounter=0
      success, myVid = myVideo.read()
      myVid = cv2.resize(myVid, (wT,hT))

      imgAug = imgWebcam.copy()

      bf = cv2.BFMatcher()  # We are using knn bruteforce matcher
      matches = bf.knnMatch(des1, des2, k=2)
      good = []
      for m,n in matches:
         if m.distance < 0.75 * n.distance:
               good.append(m)
      print(len(good))
      imgFeatures = cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None, flags=2)

      ############ Homography and Matrix relationship bet query and train image #############
      if len(good) > 20:
         detection = True
         srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
         dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
         matrix, mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
         # print(matrix)
      

         ############ Finding the Bounding Box #################
         pst = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1, 1, 2)
         dst = cv2.perspectiveTransform(pst, matrix)
         img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255,0,255),3)

         ########### Augmented Image ############
         imgWrap = cv2.warpPerspective(myVid, matrix, (imgWebcam.shape[1],imgWebcam.shape[0]))

         ########### Creating the mask ###########
         maskNew = np.zeros((imgAug.shape[0],imgAug.shape[1]),np.uint8) # create blank image of Augmentation size
         cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255)) # fill the detected area with white pixels to get mask
         maskInv = cv2.bitwise_not(maskNew) # get inverse mask
         imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv) # make augmentation area black in final image
         imgAug = cv2.bitwise_or(imgWrap, imgAug)

      ######### Calling the Stacking function #############
      imgStack = stackImages(([imgWebcam,imgAug],[imgFeatures,imgTarget]),0.5)

      ########### Displaying the FPS ################
      timer = cv2.getTickCount()
      fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
      cv2.putText(imgStack,'FPS: {} '.format(int(fps)), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,20,20), 3);
      cv2.putText(imgStack,'Target Found: {} '.format(detection), (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,20,20), 3);

      

      # cv2.imshow('imgAug', imgAug)
      # cv2.imshow('imgWrap', imgWrap)
      # # cv2.imshow('img2', img2)
      # # cv2.imshow('imgFeatures', imgFeatures)

      # # cv2.imshow('ImageTarget', imgTarget)
      # cv2.imshow('WebCam', imgWebcam)
      # cv2.imshow('First Frame of video', myVid)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
         cv2.destroyAllWindows()
      else:
         cv2.imshow('All Favourable Outcomes', imgStack)
         frameCounter+=1

   
but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=web,text='Open Cam',font=('helvetica 15 bold'))
but1.place(x=5,y=150)

but2=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=webrec,text='Record Video',font=('helvetica 15 bold'))
but2.place(x=5,y=250)

but3=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=webdet,text='View AR Mode',font=('helvetica 15 bold'))
but3.place(x=5,y=350)

but4=Button(frame,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but4.place(x=210,y=470)


root.mainloop()

