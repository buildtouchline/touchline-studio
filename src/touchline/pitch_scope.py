"""Experimental grass-pitch scope; unknown is explicit and raw boxes are retained."""
import numpy as np

def estimate_pitch_mask(rgb):
    import cv2
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('RGB uint8 image required')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([25,25,25],dtype=np.uint8),np.array([75,220,255],dtype=np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),dtype=np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    if count<2:return None
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));component=(labels==largest).astype(np.uint8)*255
    area=stats[largest,cv2.CC_STAT_AREA]/mask.size
    if area<.15 or not component[int(.8*mask.shape[0]):].any():return None
    contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros_like(component);cv2.drawContours(filled,contours,-1,255,cv2.FILLED)
    return filled

def footpoint_scope(box,source_width,source_height,mask):
    if mask is None:return 'unknown'
    x1,y1,x2,y2=box
    if not all(np.isfinite(v) for v in box) or not 0<=x1<x2<=source_width or not 0<=y1<y2<=source_height:raise ValueError('Valid source pixel box required')
    x=min(mask.shape[1]-1,int((x1+x2)/2*mask.shape[1]/source_width))
    y=min(mask.shape[0]-1,int(y2*mask.shape[0]/source_height))
    radius=2;patch=mask[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
    return 'inside_estimated_pitch' if np.count_nonzero(patch)>=max(1,patch.size//3) else 'outside_estimated_pitch'
