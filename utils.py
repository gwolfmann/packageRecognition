import cv2
import numpy as np
import random as rng

rng.seed(12345)


def get4Contours(img, cThr=[30, 10], showCanny=False, minArea=2000, filter=0, draw=False):
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
    mark_points(hull, img, 4)
    uni_hull = [hull]
    cv2.drawContours(img, uni_hull, -1, 255, 2);
    cv2.imshow('united contours', img)
    cv2.waitKey(0)

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    return finalContours


def mark_points(hull, image, q):
    """plots circles of diggerent sizes and colors around points for visual examination,
    if points are ovelaping it will be reviled by different size of circels in the same place"""
    points, _, _ = hull.shape

    if points > q:
        filterPoints(hull, q)

    for i in range(q):
        r = int(np.random.randint(100, 255, 1)[0])
        g = int(np.random.randint(100, 255, 1)[0])
        b = int(np.random.randint(100, 255, 1)[0])

        cv2.circle(image, (hull[i][0][0], hull[i][0][1]), np.random.randint(10, 20, 1)[0], (r, g, b), 2)


def filterPoints(hullPoints, q):
    points, dim1, dim2 = hullPoints.shape
    print(dim1,dim2)
    myPoints = hullPoints.reshape((points, dim2))
    myPoints = myPoints[myPoints[:,1].argsort()]
    for i in range(points):
        print(i, myPoints[i])
    hullPoints = myPoints.reshape(points,dim1,dim2)
    for i in range(points):
        print(i, hullPoints[i])
    return hullPoints

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
    imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    return imgWarp


def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5
