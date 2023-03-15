import cv2
import numpy as np
import time
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


start_time = time.time()


roa=cv2.imread('Cars223.jpg') #read image
roi = cv2.bilateralFilter(roa, 11, 17, 17) #noise suppressing
grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #convert roi into gray
#blur, thresh= cv2.threshold(grey,127,255,cv2.THRESH_BINARY) #binary
thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#cv2.imshow("Contours",morph)
#cv2.waitKey(0)

Canny=cv2.Canny(grey,35,200) #apply canny
#cv2.imshow("Top 30 contours",Canny)
#cv2.waitKey(0)



#Find my contours
contours =cv2.findContours(Canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0] #simple moins de points /= none

#cv2.drawContours(roi,contours,-1,(0,255,0),2)
#cv2.imshow("Contours lettres",roi)
#cv2.waitKey(0)

k=0

for i in contours:
        epsilon = 0.05*cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i,epsilon,True)
        if len(approx) == 4 and cv2.contourArea(i)>1000 :  #minimum contour area and 4 sides contours

            found=np.copy(roi)

            #Draw rectangle on the license plate

            x, y, w, h = cv2.boundingRect(contours[k])

            # compute rotated rectangle (minimum area)
            rect = cv2.minAreaRect(contours[k])
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # draw minimum area rectangle (rotated rectangle)
            found = cv2.drawContours(found, [box], 0, (0, 25, 255), 2)
            cv2.imshow("Bounding Rectangles", found)
            #cv2.putText(found, 'Plaque', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


            top_left_x = min([box[0][0], box[1][0], box[2][0], box[3][0]])
            top_left_y = min([box[0][1], box[1][1], box[2][1], box[3][1]])
            bot_right_x = max([box[0][0], box[1][0], box[2][0], box[3][0]])
            bot_right_y = max([box[0][1], box[1][1], box[2][1], box[3][1]])

            #Text recognition

            crop = roi[top_left_y:bot_right_y, top_left_x:bot_right_x]
            number = pytesseract.image_to_string(crop, lang='eng')

            #Delete the 3rd character (circle on german license plates)

            liste = list(range(0, len(number)-1))
            number2 = ''

            for element in liste :
                if element == 2:
                    continue
                number2 = number2 + number[element]

            print(number2)

            #Display the recognized text

            cv2.putText(found, number2, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 1)
            cv2.imshow("Plaque", found)

            hist,bin = np.histogram(crop.ravel(),256,[0,255])
            #print(hist[10])

            #histogram analysis to see the proportion of black and white, is it enough to be a license plate?
            z = 0
            b = 0

            for i in range(11):
                z = z + hist[i]

            for i in range(11, 256):
                b = b + hist[i]

            fin = b / (z + b)
            #print(fin)

            #if not enough letters recognized, go to next contour
            if len(number2)<6:
                continue

        k=k+1


#crop the license plate in the image and find each letter

gui = crop.copy()

boule = Canny[top_left_y:bot_right_y, top_left_x:bot_right_x]
cr = cv2.findContours(boule, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

m = 0
phrase = ""


for i in cr:
    if cv2.contourArea(i) >10 :

        #cv2.drawContours(gui, cr[m], -1, (0, 255, 0), 2)


        rect = cv2.minAreaRect(cr[m])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        top_left_x = min([box[0][0], box[1][0], box[2][0], box[3][0]])
        top_left_y = min([box[0][1], box[1][1], box[2][1], box[3][1]])
        bot_right_x = max([box[0][0], box[1][0], box[2][0], box[3][0]])
        bot_right_y = max([box[0][1], box[1][1], box[2][1], box[3][1]])

        crop = gui[top_left_y:bot_right_y, top_left_x:bot_right_x]
        #cv2.imshow("Paco", crop)
        #cv2.waitKey(0)
    m = m+1

cv2.imshow("Placo", gui)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(phrase)
print("--- %s seconds ---" % (time.time() - start_time))
