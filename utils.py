import cv2
import numpy as np
import random as rng

from main import hP

rng.seed(12345)


def getBaseContours(img, cThr=[30, 10], showCanny=False, minArea=2000, draw=False):
    return getImageContours(img, 4, cThr, showCanny, minArea, draw)


def getBoxContours(img, cThr=[30, 10], showCanny=False, minArea=2000, draw=False):
    return getImageContours(img, 6, cThr, showCanny, minArea, draw, False)


def getImageContours(img, qPoints, cThr=[30, 10], showCanny=False, minArea=2000, draw=False, mustFilter=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((2, 2))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThresh = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny:
        cv2.imshow('Canny', imgThresh)
        cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = np.vstack(contours[i] for i in range(len(contours)))
    finalContours = []
    hull_list = []
    for i in contours:
        peri = cv2.arcLength(i, True)
        area = cv2.contourArea(i)
        print(area, minArea)
        approx = cv2.approxPolyDP(i, 0.2 * peri, True)
        bbox = cv2.boundingRect(approx)
        finalContours.append([len(approx), area, approx, bbox, i])
        hull = cv2.convexHull(i)
        hull_list.append(hull)
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
#    for con in finalContours:
#        cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

#    for i in range(len(hull_list)):
#        x, y, w, h = cv2.boundingRect(hull_list[i])
#        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
#        cv2.drawContours(img, hull_list, i, color)
#    cv2.imshow('counterned', img)
#    cv2.waitKey(0)

    # filter down irreleveant poits
    cnt_len = cv2.arcLength(cont, True)
    cont = cv2.approxPolyDP(cont, .01 * cnt_len, True)
    hull = cv2.convexHull(cont)
    # higlight points

    if mustFilter:
        hull = filterPoints(hull, qPoints)

    uni_hull = [hull]
    if draw:
        cv2.drawContours(img, uni_hull, -1, 255, 2);
        cv2.imshow('united contours', img)
        cv2.waitKey(0)

    return hull, finalContours


def mark_points(hull, image, q):
    """plots circles of diggerent sizes and colors around points for visual examination,
    if points are ovelaping it will be reviled by different size of circels in the same place"""
    points = min(q,len(hull))
    for i in range(points):
        r = int(np.random.randint(100, 255, 1)[0])
        g = int(np.random.randint(100, 255, 1)[0])
        b = int(np.random.randint(100, 255, 1)[0])

        cv2.circle(image, (hull[i][0][0], hull[i][0][1]), np.random.randint(10, 20, 1)[0], (r, g, b), 2)
    return image


def filterPoints(hullPoints, q):
    points, dim1, dim2 = hullPoints.shape

    if points > q:
        myPoints = hullPoints.reshape((points, dim2))
        diffPoints = reorderPoints(myPoints, q)
        hullPoints = diffPoints.reshape(q, dim1, dim2)
    return hullPoints


def getLowerPoint(hullPoints):
    lowerPoint = hullPoints[0]
    for p in hullPoints:
        if p[0][1] > lowerPoint[0][1]:
            lowerPoint = p
    return lowerPoint


def getLeftPoint(hullPoints, lowerPoint):
    leftPoint = lowerPoint.copy()
    leftPoint[0][1] = 0
    leftPoint[0][0] = 0
    for p in hullPoints:
        if (not (p == lowerPoint).all()) \
                and (p[0][1] < lowerPoint[0][1]) \
                and (p[0][0] < lowerPoint[0][0]) \
                and (p[0][1] > leftPoint[0][1]):
            leftPoint = p
    return leftPoint


def getRightPoint(hullPoints, lowerPoint):
    rightPoint = lowerPoint.copy()
    rightPoint[0][1] = 0
    rightPoint[0][0] = hP
    for p in hullPoints:
        if (not (p == lowerPoint).all()) \
                and (p[0][1] < lowerPoint[0][1]) \
                and (p[0][0] > lowerPoint[0][0]) \
                and (p[0][1] > rightPoint[0][1]):
            rightPoint = p

    return rightPoint


def get3Points(hullPoints):
    points, dim1, dim2 = hullPoints.shape

    myPoints = hullPoints.reshape((points, dim2))
    lowerPoint = getLowerPoint(hullPoints)
    leftPoint = getLeftPoint(hullPoints, lowerPoint)
    rightPoint = getRightPoint(hullPoints, lowerPoint)
    tp = np.array([leftPoint, lowerPoint, rightPoint])
    tp = tp.reshape(3, dim1, dim2)
    return tp


def getContours(img, cThr=[30, 10], showCanny=False, minArea=2000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((2, 2))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThresh = cv2.erode(imgDial, kernel, iterations=2)
    if showCanny:
        cv2.imshow('Canny', imgThresh)
        cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    hull_list = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            print(area, minArea)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) >= filter:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])
            hull = cv2.convexHull(i)
            hull_list.append(hull)

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    if draw:
        print("finalcontours len=", len(finalContours))
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
        for i in range(len(hull_list)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(img, hull_list, i, color)

        cv2.imshow('counterned', img)
        cv2.waitKey(0)

    return img, finalContours


def reorderPoints(myPoints, q):
    if q == 4:
        return reorder4Points(myPoints)
    else:
        return reorder6Points(myPoints)


def reorder4Points(myPoints):
    # print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints[0:4])
    #myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def reorder6Points(myPoints):
    # print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints[0:6])
    #tomo los 3 mayores y los 3 menores
    tmppos = [0,0]
    tmpval = [0,0]
    add = myPoints.sum(1)
    maxs = np.argmax(add)
    mins = np.argmin(add)
    myPointsNew[0] = myPoints[mins]
    tmppos[0] = mins
    tmpval[0] = add[mins]
    add[mins] = 100000
    mins = np.argmin(add)
    myPointsNew[1] = myPoints[mins]
    tmppos[1] = mins
    tmpval[1] = add[mins]
    add[mins] = 100000
    mins = np.argmin(add)
    myPointsNew[2] = myPoints[mins]
    add[tmppos[0]] = tmpval[0]
    add[tmppos[1]] = tmpval[1]

    myPointsNew[5] = myPoints[maxs]
    add[maxs] = 0
    maxs = np.argmax(add)
    myPointsNew[4] = myPoints[maxs]
    add[maxs] = 0
    maxs = np.argmax(add)
    myPointsNew[3] = myPoints[maxs]
    return myPointsNew


def reorder(myPoints):
    # print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def warpImg(img, points, w, h, pad=20):
    # print(points)
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp


def findDis(pts1, pts2):
    return ((pts2[0][0] - pts1[0][0]) ** 2 + (pts2[0][1] - pts1[0][1]) ** 2) ** 0.5


def pairPoints(arrPoints):
    q = len(arrPoints)
    retArra = [((arrPoints[i]), (arrPoints[(i + 1) % q]))
         for i in range(q-1)]
    return retArra
