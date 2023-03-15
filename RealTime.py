import cv2
import numpy as np
import time
import pytesseract


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def process(fps, delay, cop):
    ts = time.perf_counter()
    print(ts)
    try:
        cv2.namedWindow("Real Time Processing")
        vc = cv2.VideoCapture(1)

        if vc.isOpened():  # try to get the first frame>>>+>>>
            rval, frame = vc.read()

            while True:
                # Capture frame by frame
                rval, frame = vc.read()

                start_time = time.time()



                pframe = cv2.bilateralFilter(frame, 11, 17, 17) #noise suppressing
                grey = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY) #convert roi into gray
                #blur, thresh= cv2.threshold(grey,127,255,cv2.THRESH_BINARY) #binary


                thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

                Canny = cv2.Canny(grey, 35, 200)  # apply canny

                contours = cv2.findContours(Canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]  # simple moins de points /= none


                k = 0

                for i in contours:
                    epsilon = 0.05 * cv2.arcLength(i, True)
                    approx = cv2.approxPolyDP(i, epsilon, True)
                    if len(approx) == 4 and cv2.contourArea(i) > 1000:


                        found = np.copy(pframe)

                        x, y, w, h = cv2.boundingRect(contours[k])

                        # compute rotated rectangle (minimum area)
                        rect = cv2.minAreaRect(contours[k])
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        # draw minimum area rectangle (rotated rectangle)

                        # cv2.imshow("Bounding Rectangles", found)



                        top_left_x = min([box[0][0], box[1][0], box[2][0], box[3][0]])
                        top_left_y = min([box[0][1], box[1][1], box[2][1], box[3][1]])
                        bot_right_x = max([box[0][0], box[1][0], box[2][0], box[3][0]])
                        bot_right_y = max([box[0][1], box[1][1], box[2][1], box[3][1]])

                        crop = thresh[top_left_y:bot_right_y, top_left_x:bot_right_x]
                        number = pytesseract.image_to_string(crop, lang='eng')


                        liste = list(range(0, len(number) - 1))
                        number2 = ''

                        for element in liste:
                            if element == 2:
                                continue
                            number2 = number2 + number[element]

                        hist, bin = np.histogram(crop.ravel(), 256, [0, 255])
                        # print(hist[10])

                        z = 0
                        b = 0

                        for i in range(11):
                            z = z + hist[i]

                        for i in range(11, 256):
                            b = b + hist[i]

                        fin = b / (z + b)

                        if len(number2)>=4:
                            found = cv2.drawContours(pframe, [box], 0, (0, 25, 255), 2)
                            cv2.putText(pframe, number2, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    k = k + 1

                cv2.imshow("Real Time Processing", pframe)
                key = cv2.waitKey(20)

                # Save the frame
                diffTime = int((time.perf_counter() - ts))

                if (diffTime % (delay / 2) == 0):
                    cv2.imwrite('camScreenshot/cam' + str(diffTime) + '.png', pframe)

                # Exit interrupt
                if key == 27:  # exit on ESC
                    break
                #cv2.waitKey(1000)

            vc.release()
            cv2.destroyWindow("Real Time Processing")
    except Exception:
        print("Execution error")
    return 0

process(2,5,3)