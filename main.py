# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from io import BytesIO

import numpy as np
import cv2
import requests
from PIL import Image


def count_empty_lots(img):
    mask = cv2.imread('masks/mask.png', 0)
    r = cv2.bitwise_and(img, img, mask=mask)

    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)

    free_lots = 0
    lots = 0

    contours, hier = cv2.findContours(r, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if 200 < cv2.contourArea(contour):
            mask = np.zeros(r.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean, std = cv2.meanStdDev(r, mask=mask)
            if mean > 100:
                free_lots += 1
                cv2.drawContours(r, [contour], 0, (255, 255, 0), 2)
            else:
                cv2.drawContours(r, [contour], 0, (0, 255, 255), 2)
            lots += 1
            print(mean, std)

    cv2.putText(r, f'{free_lots} free lots of {lots}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("video", r)
    cv2.waitKey(30000)


def retrieve_img():
    ses = requests.Session()
    resp = ses.get("http://voskresenska12g:Show_Me_Camera!@176.104.29.1/ISAPI/Streaming/channels/101/picture")
    r = np.array(Image.open(BytesIO(resp.content)))
    return r


def consequtive():
    return np.mean([
        cv2.cvtColor(cv2.imread('testimages/cam1/consequtive/1.jpg'), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread('testimages/cam1/consequtive/2.jpg'), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(cv2.imread('testimages/cam1/consequtive/3.jpg'), cv2.COLOR_BGR2GRAY),
    ])


if __name__ == '__main__':
    # r = retrieve_img()
    r = cv2.imread('testimages/cam1/3.jpg')
    # r = consequtive()
    count_empty_lots(r)
