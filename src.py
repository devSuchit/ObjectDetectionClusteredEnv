#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: suchit
"""

#packages
import numpy as np
import cv2

#detecting dominant colour
def dominance(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)
    
#extracting contours around key-points
def extract_contour(a,x,y):
    x1 = int(x)
    y1 = int(y)
    crop_img = a[y1:y1+612, x1:x1+330]
    return crop_img

def main():
    #input images
    img1 = cv2.imread('photo.jpeg',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('template1.jpg',cv2.IMREAD_GRAYSCALE)
    img11 = cv2.imread('photo.jpeg')
    img22 = cv2.imread('template1.jpg')
    
    #dominant colour in template
    dominant = dominance(img22)
    
    #pre-processing
    img2 = cv2.GaussianBlur(img2,(1,1),0)
    img1 = cv2.resize(img1,None,fx=6, fy=6, interpolation = cv2.INTER_LINEAR)
    kernel2 = np.ones((15,15),np.float32)/225
    img2 = cv2.filter2D(img2,-1,kernel2)
    kernel = np.array([[0,-1,0], [-1,5.5,-1], [0,-1,0]])
    img1 = cv2.filter2D(img1, -1, kernel)

    # ORB Detector
    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    #Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    pts = np.asarray([[p.pt[0], p.pt[1]] for p in kp1])
    cols = pts[:,0]
    rows = pts[:,1]
    print(img11.shape)
    for i in range(50):
        portion = extract_contour(img11,cols[i]/6,rows[i]/6)
        temp = dominance(portion)
        if (abs(sum(temp)-sum(dominant))<6):
            cv2.rectangle(matching_result,(int(cols[i]),int(rows[i])),(int(cols[i])+612,int(rows[i])+330),(255,0,0),20)
            cv2.putText(matching_result, "object", (int(cols[i]), int(rows[i]-2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3.0, (0,255,0), 2);
    
    #output
    cv2.imwrite("Matchingresult.jpg", matching_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
