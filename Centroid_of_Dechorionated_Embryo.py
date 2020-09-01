# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:48:54 2019

@author: enet-joshi317-admin
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import pi,cos,sin

img=cv2.imread('ir_cropped_dechorionated_image_2.jpg',0)
#thresh,img_new = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
#thresh,img_new = cv2.threshold(img,0,43,cv2.THRESH_OTSU)
# define coordinates
#y1_4x_box_new=0
#crop_img_new = img[y1_4x_box_new:y2_4x_box_new,x1_4x_box_new:x2_4x_box_new]

edges = cv2.Canny(img,40,80)

edges_nonub = cv2.Canny(img,40,150)

edges1 = cv2.Canny(img,10,45)
#filters out small images taken from image 
kernel = np.ones((10,10),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cnts_old, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts_old_nonub, _ = cv2.findContours(edges_nonub.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts_edges1, _ = cv2.findContours(edges1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

moments_old  = [cv2.moments(cnt) for cnt in cnts_old]
for i in range(len(moments_old)):
    if moments_old[i]['m00']==0:
        moments_old[i]['m00']=1
centroids_old = [( int(m['m10']/m['m00']),int(m['m01']/m['m00']) ) for m in moments_old]
cX_old = [c[0] for c in centroids_old]
#print ('X coordinates of centroids original:', cX_old)
cY_old = [c[1] for c in centroids_old]
#print ('Y coordinates of centroids original:', cY_old)


contours_reshape_old=[]
contour_lengths_old=[]
contour_areas_old=[]
for e_old in range(len(cnts_old)):
    contours_reshape_old.append(cnts_old[e_old].reshape(-1,2))
    contour_lengths_old.append(cv2.arcLength(contours_reshape_old[e_old],True))
    contour_areas_old.append(cv2.contourArea(contours_reshape_old[e_old],True))
    e_old+=1
indices=[u for u, c in enumerate(contour_lengths_old) if c>=max(contour_lengths_old)-1]

largest_contours_x_old=[]
largest_contours_y_old=[]
for f in range(len(cnts_old[indices[0]])):
    largest_contours_x_old.append(cnts_old[indices[0]][f][0][0])
    largest_contours_y_old.append(cnts_old[indices[0]][f][0][1])
    
contours_reshape_old_nonub=[]
contour_lengths_old_nonub=[]
contour_areas_old_nonub=[]
for e_old in range(len(cnts_old_nonub)):
    contours_reshape_old_nonub.append(cnts_old_nonub[e_old].reshape(-1,2))
    contour_lengths_old_nonub.append(cv2.arcLength(contours_reshape_old_nonub[e_old],True))
    contour_areas_old_nonub.append(cv2.contourArea(contours_reshape_old_nonub[e_old],True))
    e_old+=1
indices=[u for u, c in enumerate(contour_lengths_old_nonub) if c>=max(contour_lengths_old_nonub)-1]

largest_contours_x_old=[]
largest_contours_y_old=[]
for alll in range(len(cnts_old)):
    for j_new in range(len(cnts_old[alll])):
        largest_contours_x_old.append((cnts_old[alll][j_new][0][0]))
        largest_contours_y_old.append((cnts_old[alll][j_new][0][1]))

largest_contours_x_edges1=[]
largest_contours_y_edges1=[]
for alll in range(len(cnts_edges1)):
    for j_new in range(len(cnts_edges1[alll])):
        largest_contours_x_edges1.append((cnts_edges1[alll][j_new][0][0]))
        largest_contours_y_edges1.append((cnts_edges1[alll][j_new][0][1]))
    
rect = cv2.minAreaRect(cnts_old_nonub[indices[0]])
box = cv2.boxPoints(rect)
box = np.int0(box)
y_embryo_=abs(round((((box[0][0]-box[1][0])**2)+((box[0][1]-box[1][1])**2))**.5))
x_embryo_=abs(round((((box[3][0]-box[0][0])**2)+((box[0][1]-box[3][1])**2))**.5))
if y_embryo_>x_embryo_:
    y_embryo=y_embryo_
else:
    y_embryo=x_embryo_
    
angless_new=np.linspace(0,360,361)
def contour_calc(angle_new):
    largest_contours_x_nonub=[]
    largest_contours_y_nonub=[]
    for f in range(len(cnts_old_nonub[indices[0]])):
        largest_contours_x_nonub.append((cnts_old_nonub[indices[0]][f][0][0]*cos(angle_new*(pi/180))-cnts_old_nonub[indices[0]][f][0][1]*sin(angle_new*(pi/180))))
        largest_contours_y_nonub.append((cnts_old_nonub[indices[0]][f][0][0]*sin(angle_new*(pi/180))+cnts_old_nonub[indices[0]][f][0][1]*cos(angle_new*(pi/180))))
        max_y_new=max(largest_contours_y_nonub)
        max_y_new_index=[u for u, c in enumerate(largest_contours_y_nonub) if c==max_y_new]
    return largest_contours_x_nonub,largest_contours_y_nonub,max_y_new,max_y_new_index
for angless in range(len(angless_new)):
    contour_calc(angless_new[angless])
    if np.isin(abs(round(contour_calc(angless_new[angless])[0][contour_calc(angless_new[angless])[3][0]])+(min(largest_contours_x_old))),range(cX_old[0]-5,cX_old[0]+5)) and np.isin(round(abs(max(contour_calc(angless_new[angless])[1])-min(contour_calc(angless_new[angless])[1]))),range(int(y_embryo)-5,int(y_embryo)+5)):
        print ('first angle:',angless_new[angless])
        break
    else:
        continue
largest_contours_x_nonub=contour_calc(45)[0]
largest_contours_y_nonub=contour_calc(45)[1]

angless_new_old=np.linspace(0,360,361)

def contour_calc_new_old(angle_new):
    largest_contours_new_old=[]
    largest_contours_x_new_old=[]
    largest_contours_y_new_old=[]
    for f_new in range(len(largest_contours_x_old)):
        largest_contours_x_new_old.append((largest_contours_x_old[f_new]*cos(angle_new*(pi/180))-largest_contours_y_old[f_new]*sin(angle_new*(pi/180))))
        largest_contours_y_new_old.append((largest_contours_x_old[f_new]*sin(angle_new*(pi/180))+largest_contours_y_old[f_new]*cos(angle_new*(pi/180))))
    for q_new in range(len(largest_contours_x_old)):
        largest_contours_new_old.append([[largest_contours_x_new_old[q_new],largest_contours_y_new_old[q_new]]])
    largest_contours_new_old=np.asarray(largest_contours_new_old,dtype='int32')
    rect_new_old = cv2.minAreaRect(largest_contours_new_old)
    box_new_old = cv2.boxPoints(rect_new_old)
    box_new_old = np.int0(box_new_old)
    return largest_contours_x_new_old,largest_contours_y_new_old,largest_contours_new_old,box_new_old

largest_contours_x_new_old=contour_calc_new_old(45)[0]
largest_contours_y_new_old=contour_calc_new_old(45)[1]

angless_new_old=np.linspace(0,360,361)

def contour_calc_new_old_edges1(angle_new):
    largest_contours_new_edges1=[]
    largest_contours_x_new_edges1=[]
    largest_contours_y_new_edges1=[]
    for f_new in range(len(largest_contours_x_edges1)):
        largest_contours_x_new_edges1.append((largest_contours_x_edges1[f_new]*cos(angle_new*(pi/180))-largest_contours_y_edges1[f_new]*sin(angle_new*(pi/180))))
        largest_contours_y_new_edges1.append((largest_contours_x_edges1[f_new]*sin(angle_new*(pi/180))+largest_contours_y_edges1[f_new]*cos(angle_new*(pi/180))))
    for q_new in range(len(largest_contours_x_old)):
        largest_contours_new_edges1.append([[largest_contours_x_new_edges1[q_new],largest_contours_y_new_edges1[q_new]]])
    largest_contours_new_edges1=np.asarray(largest_contours_new_edges1,dtype='int32')
    rect_new_old = cv2.minAreaRect(largest_contours_new_edges1)
    box_new_old = cv2.boxPoints(rect_new_old)
    box_new_old = np.int0(box_new_old)
    return largest_contours_x_new_edges1,largest_contours_y_new_edges1,largest_contours_new_edges1,box_new_old

largest_contours_x_new_edges1=contour_calc_new_old_edges1(45)[0]
largest_contours_y_new_edges1=contour_calc_new_old_edges1(45)[1]

x1_germ_box=min(largest_contours_x_nonub)+10
x2_germ_box=max(largest_contours_x_nonub)-10
y1_germ_box=min(largest_contours_y_nonub)
y2_germ_box=min(largest_contours_y_nonub)+20

angle_rot=315

x1_germ_box_rotated=x1_germ_box*cos(angle_rot*(pi/180))-y1_germ_box*sin(angle_rot*(pi/180))
y1_germ_box_rotated=x1_germ_box*sin(angle_rot*(pi/180))+y1_germ_box*cos(angle_rot*(pi/180))

x2_germ_box_rotated=x2_germ_box*cos(angle_rot*(pi/180))-y1_germ_box*sin(angle_rot*(pi/180))
y2_germ_box_rotated=x2_germ_box*sin(angle_rot*(pi/180))+y1_germ_box*cos(angle_rot*(pi/180))

x3_germ_box_rotated=x1_germ_box*cos(angle_rot*(pi/180))-y2_germ_box*sin(angle_rot*(pi/180))
y3_germ_box_rotated=x1_germ_box*sin(angle_rot*(pi/180))+y2_germ_box*cos(angle_rot*(pi/180))

x4_germ_box_rotated=x2_germ_box*cos(angle_rot*(pi/180))-y2_germ_box*sin(angle_rot*(pi/180))
y4_germ_box_rotated=x2_germ_box*sin(angle_rot*(pi/180))+y2_germ_box*cos(angle_rot*(pi/180))
if max(largest_contours_y_new_old)>max(largest_contours_y_nonub):
    print('Germcell on Lower y-axis')
else:
    print('Germcell on Upper y-axis')

## Crop Embryo Germcell
#crop_img_germcell = img[int(y1_germ_box_rotated):int(y2_germ_box_rotated),int(x1_germ_box_rotated):int(x2_germ_box_rotated)]
crop_img_germcell = img[int(y1_germ_box_rotated):-1:int(y2_germ_box_rotated),int(x1_germ_box_rotated):-1:int(x2_germ_box_rotated)]



plt.figure(1)
plt.title('First Edge Detection Dechorionated Embryo')
plt.xlabel('x coordinate (px)')
plt.ylabel('y coordinate (px)')
plt.imshow(edges,cmap = 'gray')

plt.figure(2)
plt.title('First Edge Detection Dechorionated Embryo No Nub')
plt.xlabel('x coordinate (px)')
plt.ylabel('y coordinate (px)')
plt.imshow(edges_nonub,cmap = 'gray')

plt.figure(3)
plt.title('Germcell Line First Edge Detection Dechorionated Embryo')
plt.xlabel('x coordinate (px)')
plt.ylabel('y coordinate (px)')
plt.imshow(edges1,cmap = 'gray')

"""
plt.figure(4)
plt.title('Rotated Dechorionated Embryo No Nub')
plt.xlabel('x coordinate (px)')
plt.ylabel('y coordinate (px)')
plt.plot(largest_contours_x_nonub,largest_contours_y_nonub,'r')

plt.figure(5)
plt.title('Rotated Dechorionated Embryo')
plt.xlabel('x coordinate (px)')
plt.ylabel('y coordinate (px)')
plt.plot(largest_contours_x_new_old,largest_contours_y_new_old,'r')

plt.figure(6)
plt.title('Rotated Germcell Line Embryo')
plt.xlabel('x coordinate (px)')
plt.ylabel('y coordinate (px)')
plt.plot(largest_contours_x_new_edges1,largest_contours_y_new_edges1,'r')
"""

plt.figure(4)
plt.title('Rotated Germcell Line Embryo Box')
plt.xlabel('x coordinate (px)')
plt.ylabel('y coordinate (px)')
plt.plot(x1_germ_box_rotated,y1_germ_box_rotated,'bo'),plt.imshow(edges1,cmap = 'gray')
plt.plot(x2_germ_box_rotated,y2_germ_box_rotated,'bo'),plt.imshow(edges1,cmap = 'gray')
plt.plot(x3_germ_box_rotated,y3_germ_box_rotated,'bo'),plt.imshow(edges1,cmap = 'gray')
plt.plot(x4_germ_box_rotated,y4_germ_box_rotated,'bo'),plt.imshow(edges1,cmap = 'gray')
#plt.plot(x1_germ_box_rotated,y1_germ_box_rotated,'bo')
#plt.plot(x2_germ_box_rotated,y2_germ_box_rotated,'bo')
#plt.plot(x3_germ_box_rotated,y3_germ_box_rotated,'bo')
#plt.plot(x4_germ_box_rotated,y4_germ_box_rotated,'bo')
#plt.plot(largest_contours_x_nonub,largest_contours_y_nonub,'r')

#plt.figure(8)
#plt.title('Cropped Germcell Line Embyro Image')
#plt.xlabel('x coordinate (px)')
#plt.ylabel('y coordinate (px)')
#plt.imshow(crop_img_germcell,cmap = 'gray')
#plt.show()
