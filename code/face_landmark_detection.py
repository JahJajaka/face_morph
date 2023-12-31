import sys
import os
import dlib
import glob
import numpy as np
from skimage import io
import cv2
from imutils import face_utils
DEFECTED = ['165.jpg', '177.jpg', '185.jpg', '186.jpg', '190.jpg', '199.jpg', '202.jpg', '206.jpg']
class NoFaceFound(Exception):
   """Raised when there is no face found"""
   pass

def calculate_margin_help(img1,img2):
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0]-size2[0])//2
    diff1 = abs(size1[1]-size2[1])//2
    avg0 = (size1[0]+size2[0])//2
    avg1 = (size1[1]+size2[1])//2

    return [size1,size2,diff0,diff1,avg0,avg1]

def crop_image(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)
    
    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        #print("la")
        scale0 = size1[0]/size2[0]
        scale1 = size1[1]/size2[1]

        if(scale0 > scale1):
            res = cv2.resize(img2,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img2,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(img1,res)

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        #print("ici")
        scale0 = size2[0]/size1[0]
        scale1 = size2[1]/size1[1]
        if(scale0 > scale1):
            res = cv2.resize(img1,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img1,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(res,img2)

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        #print("ou alors")
        return [img1[diff0:avg0,:],img2[:,abs(diff1):avg1]]
    
    else:
        #print("ou bien")
        return [img1[:,diff1:avg1],img2[abs(diff0):avg0,:]]

def crop_image_help(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)
    #print(size1,size2,diff0,diff1,avg0,avg1)
    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        #print("ici 1")
        return [img1,img2[abs(diff0):avg0,abs(diff1):avg1]]

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        #print("ici 2")
        return [img1[diff0:avg0,diff1:avg1],img2]

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        #print("ici 3")
        return [img1[diff0:avg0,:],img2[:,abs(diff1):avg1]]

    else:
        #print("ici 4")
        return [img1[:,diff1:avg1],img2[abs(diff0):avg0,:]]

def generate_face_correspondences(theImage1, theImage2, img_paths):
    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('code/utils/shape_predictor_68_face_landmarks.dat')
    corresp = np.zeros((68,2))

    imgList = crop_image(theImage1,theImage2)
    list1 = []
    list2 = []
    j = 1
    
    for img in imgList:
        #cv2.imshow("image courrante", img)
        #cv2.waitKey(0)
        size = (img.shape[0],img.shape[1])
        if(j == 1):
            currList = list1
        else:
            currList = list2

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        #print(img.shape)
        dets = detector(img, 2)
        
        print(f'dets:{len(dets)}')
        # try:
        #     if len(dets) == 0:
        #         raise NoFaceFound
        # except NoFaceFound:
        #     print("Sorry, but I couldn't find a face in the image.")
        #     #cv2.imshow("no face", img)
        #     #cv2.waitKey(0)
            

        j=j+1
        if len(dets) == 0:

            for img_path in img_paths:
                if img_path in DEFECTED:
                    print(img_path)
                    arr = np.genfromtxt(f"165.csv",dtype=int, delimiter=',')           
                    for i in range(0,68):
                        x = arr[i][0]
                        y = arr[i][1]
                        print(f'x:{x},y:{y}')
                        currList.append((x, y))
                        corresp[i][0] += x
                        corresp[i][1] += y
                        #cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

                    # Add back the background
                    currList.append((1,1))
                    currList.append((size[1]-1,1))
                    currList.append(((size[1]-1)//2,1))
                    currList.append((1,size[0]-1))
                    currList.append((1,(size[0]-1)//2))
                    currList.append(((size[1]-1)//2,size[0]-1))
                    currList.append((size[1]-1,size[0]-1))
                    currList.append(((size[1]-1),(size[0]-1)//2))
        else:
            for k, rect in enumerate(dets[:1]):
                print(f'rect:{rect}')
                (x1, y1, w, h) = face_utils.rect_to_bb(rect)
                print(f'rect first:{x1}')
                if x1 < 0:
                    rect = dlib.grow_rect(rect, x1-100)
                    print(f'grown rect: {rect}')
                # Get the landmarks/parts for the face in rect.
                shape = predictor(img, rect)
                # corresp = face_utils.shape_to_np(shape)
                
                for i in range(0,68):
                    x = shape.part(i).x
                    y = shape.part(i).y
                    currList.append((x, y))
                    corresp[i][0] += x
                    corresp[i][1] += y
                    #cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

                # Add back the background
                currList.append((1,1))
                currList.append((size[1]-1,1))
                currList.append(((size[1]-1)//2,1))
                currList.append((1,size[0]-1))
                currList.append((1,(size[0]-1)//2))
                currList.append(((size[1]-1)//2,size[0]-1))
                currList.append((size[1]-1,size[0]-1))
                currList.append(((size[1]-1),(size[0]-1)//2))
        
    # Add back the background
    narray = corresp/2
    narray = np.append(narray,[[1,1]],axis=0)
    narray = np.append(narray,[[size[1]-1,1]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,1]],axis=0)
    narray = np.append(narray,[[1,size[0]-1]],axis=0)
    narray = np.append(narray,[[1,(size[0]-1)//2]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,size[0]-1]],axis=0)
    narray = np.append(narray,[[size[1]-1,size[0]-1]],axis=0)
    narray = np.append(narray,[[(size[1]-1),(size[0]-1)//2]],axis=0)
    #cv2.imwrite(f'result1.jpg', imgList[0])
    #cv2.imwrite(f'result2.jpg', imgList[1])
    #cv2.imwrite(f'result3.jpg', narray)
    return [size,imgList[0],imgList[1],list1,list2,narray]