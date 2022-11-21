import cv2
import numpy as np
import time

def process(fps,delay, cop):
    ts=time.perf_counter()
    print(ts)
    try:
        cv2.namedWindow("Real Time Processing")
        vc = cv2.VideoCapture(0)
        
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
    
            while True:
                #Capture frame by frame
                rval, frame = vc.read()

                #Processing operation
                    #cv2.threshold(src, threshold, maxValue, type)
                if (cop==1):
                    # Any value greater than the threshold thresh is replaced with maxval and the other values are replaced with 0
                    (T,pframe) = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)  
                elif (cop==2):
                    # The value greater than the threshold thresh remains the same, and the other values are replaced with 0
                    (T,pframe) = cv2.threshold(frame, 128, 255, cv2.THRESH_TOZERO) 
                elif (cop==3):
                    #Convert to pframescale 
                    noisereduc=cv2.bilateralFilter(frame,11,17,17)
                    pframe = cv2.cvtColor(noisereduc, cv2.COLOR_BGR2GRAY)
                    #Threshold process
                    (T,pframe) = cv2.threshold(pframe, 128, 255, cv2.THRESH_BINARY)
                    pframe=cv2.Canny(pframe,35,200)
                    #Autobrightness
                    pframe=autoBrightness(pframe)
                elif (cop==4):
                    #Convert to pframescale 
                    pframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #Threshold process
                    (T,pframe) = cv2.threshold(pframe, 128, 255, cv2.THRESH_TOZERO) 
                    #Autobrightness
                    pframe=autoBrightness(pframe)
                else:
                    print("dumb man")
                    break
                
                cv2.imshow("Real Time Processing", pframe)
                key = cv2.waitKey(20)
                
                #Save the frame
                diffTime=int((time.perf_counter()-ts))
                
                if (diffTime%(delay/2)==0):
                    cv2.imwrite('camScreenshot/cam'+str(diffTime)+'.png',pframe)
                    
                #Exit interrupt
                if key == 27: # exit on ESC
                    break
        
        vc.release()
        cv2.destroyWindow("Real Time Processing")
    except Exception:
        print("Execution error")
    return 0

def autoBrightness(img):
    cols,rows=img.shape
    brightness=np.sum(img)/(255*cols*rows)
    minimum_brightness = 0.66
    ratio = brightness / minimum_brightness
    if ratio >= 1:
        return img

    # Otherwise, adjust brightness to get the target brightness
    return cv2.convertScaleAbs(img, alpha = 1 / ratio, beta = 0)
    

process(15,5,3)