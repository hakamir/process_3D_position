
"""
Created on Mon Jun  3 17:22:26 2019

@author: antoine
"""

import cv2
import os
import numpy as np
from scipy import ndimage

def segmentation(bbox,disparity,image):
    """
    Input:  Detection bounding box :[x1,y1,x2,y2,...]
            Disparity of the frame :array of uint16 the same size as the image
            image path

    Output: Tuple with the mask of the object and the new bounding box: (mask,(x1,y1,x2,y2))
            new image with the mask as overlay
    """

    masks=[]
    # we compute the mask and new bounding box
    x1,y1,x2,y2=bbox

    #read the disparity of the image in the detection bounding box and convert it to uint8
    disp_seg=disparity[y1:y2,x1:x2]
    disp_seg=np.array(disp_seg,dtype=np.float32)
    disp_seg/=257
    disp_seg=np.array(disp_seg,dtype=np.uint8)

    # create the mask by segmenting the disparity with Otsu
    segmented_image=cv2.threshold(disp_seg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    #get the first and last indx of non-zero values on the x axis at mid height
    #to get the new x1 and x2 (we keep the same y1 and y2 because we don't care about y axis too much)
    non_zero_indx=np.nonzero(segmented_image[int(len(segmented_image)/2),:])[0]
    if len(non_zero_indx)==0:
        new_x1=x1
        new_x2=x2
    else:
        new_x1=x1+non_zero_indx[0]
        new_x2=x1+non_zero_indx[-1]

    #create a the new image with the overlay on
    mask_resized=np.zeros(np.shape(image),np.uint8)
    mask_resized[y1:y2,x1:x2,0]=segmented_image

    alpha=0.1
    cv2.addWeighted(mask_resized, alpha, image, 1 - alpha,0, image)


    masks.append((segmented_image,new_x1,y1,new_x2,y2))
    masks=np.array(masks)
    return masks,image
