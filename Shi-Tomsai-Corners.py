'''
Code for Shi-Tomsai Corners(Modified form of Harris corners)
Author: hasanlatif.pk@gmail.com
'''


import numpy as np
import cv2
from scipy import signal
import numpy.linalg as LA


img = cv2.imread('test.jpg',0)   # reading image in gray scale mode,'0' flag mean reading image in grayscale mode
new_img = np.zeros((img.shape[0],img.shape[1]))  ## making new image of the same shape as of orignal image
img = np.float32(img)/255       # Normalizing imaging to make values in bound
I_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)  ## Applying First dervative along x-direction
I_y = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=1) ### Applying first derivative aling y-direction
I_x_2 = I_x*I_x   ## element of structed matrixx
I_y_2 = I_y*I_y   ## Element of Structed matrix
I_xy = I_x*I_y    ## Element of structured matrix
I_yx = I_y*I_x   ## Element of Structed matrix
kernel = np.ones((3,3))/9   ## As we are using weight as 1.so defining kernal as 1 and as element in each window is 9. . Simply using box filter does the same
I_x_sum = signal.convolve2d(I_x,kernel,mode='same')
I_x_2_sum =signal.convolve2d(I_x_2,kernel,mode='same')
I_y_sum =signal.convolve2d(I_y,kernel,mode='same')
I_y_2_sum = signal.convolve2d(I_y_2,kernel,mode='same')
I_xy_sum = signal.convolve2d(I_xy,kernel,mode='same')
I_yx_sum = signal.convolve2d(I_yx,kernel,mode='same')
'''
print("[Info] I_x_2_sum",I_x_2_sum)
print("[Info] I_y_2_sum",I_y_2_sum.shape)
print("[Info] I_xy_sum",I_xy_sum.shape)
print("[info],I_yx",I_yx.shape)
'''
list1=[]
for row in range(0,img.shape[0]):
    for col in range(0,img.shape[1]):
        Hessian_matrix = np.array([[I_x_sum[row][col] , I_xy_sum[row][col]],[I_yx_sum[row][col], I_y_2_sum[row][col]]])  ##Finding Hessian Matrx and its eigen value
        eigen_value=LA.eigvals(Hessian_matrix)  # Eigen value
        R = np.min(eigen_value)   ## SHi-tomsai criteria
        if (R>img[row][col]):   ## If score is bigger then that pixel is image
            new_img[row][col]=img[row][col]


cv2.putText(new_img,"Detected Corners",(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
cv2.putText(img,"original Image",(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
cv2.line(img,(0,0),(0,511),(255,0,0),5)
cv2.imshow("Detected Corners + Original image",np.hstack((new_img*255,img)))
cv2.waitKey(0)
