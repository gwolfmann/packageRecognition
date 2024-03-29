import cv2
import numpy
import numpy as np

import utils
# This is a sample Python script.

########
web_cam = False
image1 = './images/toma11_1.png' ##1200*901
image2 = './images/toma11_2.png' ##1200*901
image3 = './images/toma11_3.png' ##1200*901
#image3 = './images/toma12_2.jpg'
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)
scale = 1
wP = 1500 * scale # el original fue 210 * scale
hP = 2250 * scale # 297 * scale
########

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# set the lower and upper bounds for the green hue
lower_green = np.array([50, 100, 50])
upper_green = np.array([70, 255, 255])
lower_red = np.array([160, 100, 50])
upper_red = np.array([180, 255, 255])
lower_yel = np.array([15, 50, 0])
upper_yel = np.array([35, 255, 255])
lower_white = np.array([0, 0, 60])
upper_white = np.array([107, 111, 62])
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def show_image():
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    img3 = cv2.imread(image3)

    img1 = cv2.resize(img1, (0, 0), None, 0.5, 0.5)
    img2 = cv2.resize(img2, (0, 0), None, 0.5, 0.5)
    img3 = cv2.resize(img3, (0, 0), None, 0.5, 0.5)

    cv2.imshow('Original_1', img1)
    cv2.imshow('Original_2', img2)
    cv2.imshow('Original_3', img3)
    cv2.waitKey(0)

    # convert the BGR image to HSV colour space
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

    ## create a mask for green colour using inRange function
    ymask = cv2.inRange(hsv, lower_yel, upper_yel)
    #rmask = cv2.inRange(hsv, lower_red, upper_red)
    ##mask = cv2.bitwise_or(ymask, rmask)
    mask = ymask

    #cv2.imshow('yell_mask', mask)
    res1 = cv2.bitwise_and(img1, img1, mask=mask)
    #cv2.imshow('Original_yel1', res1)

    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    ymask = cv2.inRange(hsv, lower_blue, upper_blue)
    #rmask = cv2.inRange(hsv, lower_red, upper_red)
    #mask = cv2.bitwise_or(ymask, rmask)
    mask = ymask
    res2 = cv2.bitwise_or(img1, img1, mask=mask)
    cv2.imshow('Original_blue', res2)
    cv2.waitKey(0)

    hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    ymask = cv2.inRange(hsv, lower_yel, upper_yel)
    rmask = cv2.inRange(hsv, lower_red, upper_red)
    #mask = cv2.bitwise_or(ymask, rmask)
    mask = ymask
    #mask = cv2.inRange(hsv, lower_red, upper_red)
    res3 = cv2.bitwise_and(img3, img3, mask=mask)
    cv2.imshow('Original_yel3', res3)

    utils.getContours(res1, showCanny=True, draw=True)
    cv2.waitKey(0)
    utils.getContours(res2, showCanny=True, draw=True)
    cv2.waitKey(0)
    utils.getContours(res3, showCanny=True, draw=True)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def originalSizing():
    img = cv2.imread(image3)
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    cv2.imshow('Original', img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ymask = cv2.inRange(hsv, lower_yel, upper_yel)
    wmask = cv2.inRange(hsv, lower_white, upper_white)
    bmask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.bitwise_or(ymask, bmask)
    #mask = bmask
    #cv2.imshow('blue&yell_mask', mask)
    res1 = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('Original_yel1', res1)
    cv2.waitKey(0)

    imgContours, conts = utils.getContours(res1, minArea=20000, filter=4, showCanny=True)
    if len(conts) != 0:
        biggest = conts[0][2]
        # print(biggest)
        imgWarp = utils.warpImg(img, biggest, wP, hP)
        imgContours2, conts2 = utils.getContours(imgWarp, showCanny=True,
                                                 minArea=2000, filter=4,
                                                 draw=True)
        #cv2.imshow('warp', imgWarp)
        #cv2.imshow('contours2', imgContours2)
        #cv2.waitKey(0)
        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = utils.reorder(obj[2])
                nW = round((utils.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
                nH = round((utils.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        imgContours2 = cv2.resize(imgContours2, (0, 0), None, 0.5, 0.5)
        cv2.imshow('A4', imgContours2)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow('Original', img)
    cv2.waitKey(0)


def filter_image(img, lower_mask, upper_mask):
    if type(img) != numpy.ndarray:
        img = cv2.imread(img)
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fmask = cv2.inRange(hsv, lower_mask, upper_mask)
    res1 = cv2.bitwise_and(img, img, mask=fmask)
    return res1


def filter_image_or(img, lower_mask1, upper_mask1, lower_mask2, upper_mask2):
    if type(img) != numpy.ndarray:
        img = cv2.imread(img)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_mask1, upper_mask1)
    mask2 = cv2.inRange(hsv, lower_mask2, upper_mask2)
    ormask = cv2.bitwise_or(mask1, mask2)
    res1 = cv2.bitwise_or(img, img, mask=ormask)
    return res1


def blue_image(image):
    return filter_image(image, lower_blue, upper_blue)


def yellow_image(image):
    return filter_image(image, lower_yel, upper_yel)


def blue_yellow_image(image):
    return filter_image_or(image, lower_blue, upper_blue, lower_yel, upper_yel)


def get_corners(image):
    img = cv2.imread(image)
    if img is None:
        print(image+" not found")
        return
    resizeFactor = 0.5
    resBlue = blue_image(image)
    #cv2.imshow('Original_blue', resBlue)
    resYell = yellow_image(image)
    #cv2.imshow('Original_yellow', resYell)
    img = cv2.resize(img, (0, 0), None, resizeFactor, resizeFactor)
    basePoints, _ = utils.getBaseContours(resBlue, minArea=20000, showCanny=False)

    #imgDoted = utils.mark_points(basePoints, img, len(basePoints))
    cv2.imshow('Original', img)
    cv2.waitKey(0)

    imgWrp = utils.warpImg(img, basePoints, wP, hP, 20)
    cv2.imshow('Original warped',  cv2.resize(imgWrp, (0, 0), None, resizeFactor, resizeFactor))
    cv2.waitKey(0)
    #imgWrp = cv2.resize(imgWrp, (0, 0), None, resizeFactor, resizeFactor)

    resYell = yellow_image(imgWrp)
    boxPoints, contsBox = utils.getBoxContours(resYell, minArea=20000, showCanny=False, draw=True)
    resBoth = blue_yellow_image(imgWrp)
    completeImg = utils.mark_points(boxPoints, resBoth, len(boxPoints))

    triPoints = utils.get3Points(boxPoints)


    pairOfPoints = utils.pairPoints(triPoints) #boxPoints

    for pointsPair in pairOfPoints:
        point1 = pointsPair[0]
        point2 = pointsPair[1]
        dist = round(utils.findDis(point1, point2) // resizeFactor // 10, 1)

#           ancho = round((utils.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
#           alto = round((utils.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)
        cv2.arrowedLine(completeImg, (point1[0][0], point1[0][1]), (point2[0][0], point2[0][1]),
                        (255, 0, 255), 3, 8, 0, 0.05)
        cv2.putText(completeImg, '{}cm'.format(dist), (point1[0][0] + 30, point1[0][1] ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                    (255, 0, 255), 2)
#           cv2.arrowedLine(imgWrp, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
#                           (255, 0, 255), 3, 8, 0, 0.05)
#           x, y, w, h = fig[3]
    cv2.imshow('Warped_doted', completeImg)
    cv2.waitKey(0)


def makeTriFiles(root, seq):
    trio = ["", "", ""]
    for i in range(3):
        file = f"{seq}_{i+1}.jpg" #= './images/toma12_2.jpg'
        trio[i] = root + file
    return trio


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    #show_image()
    #originalSizing()
    #get_corners(image3)
    trio = makeTriFiles('./images/toma', 11)
    for f in trio:
        get_corners(f)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
