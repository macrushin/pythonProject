import cv2
import numpy as np


def sift(img, cap):
    sift = cv2.SIFT_create()
    kp_image, desc_image = sift.detectAndCompute(img, None)
    bf = cv2.BFMatcher()

    while cap.isOpened():
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        kp_grayframe, desc_grayframe = sift.detectAndCompute(gray_frame, None)
        matches = bf.knnMatch(desc_image, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_points.append(m)
        cv2.namedWindow("Homography", cv2.WINDOW_NORMAL)
        if len(good_points) > 10:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            # Perspective transform
            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 255, 0), 3)
            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", gray_frame)
        key = cv2.waitKey(1)
        if key == 1:
            break
    cap.release()
    cv2.destroyAllWindows()


def orb(img, cap):
    orb = cv2.ORB_create()
    kp_image, desc_image = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()

    while cap.isOpened():
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        kp_grayframe, desc_grayframe = orb.detectAndCompute(gray_frame, None)
        matches = bf.knnMatch(desc_image, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_points.append(m)
        cv2.namedWindow("Homography", cv2.WINDOW_NORMAL)
        if len(good_points) > 10:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            # Perspective transform
            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 255, 0), 3)
            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", gray_frame)
        key = cv2.waitKey(1)
        if key == 1:
            break
    cap.release()
    cv2.destroyAllWindows()

def surf(img, cap):
    surf = cv2.xfeatures2d_SURF
    kp_image, desc_image = surf.detectAndCompute(img, None)
    bf = cv2.BFMatcher()

    while cap.isOpened():
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        kp_grayframe, desc_grayframe = surf.detectAndCompute(gray_frame, None)
        matches = bf.knnMatch(desc_image, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_points.append(m)
        cv2.namedWindow("Homography", cv2.WINDOW_NORMAL)
        if len(good_points) > 10:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            # Perspective transform
            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 255, 0), 3)
            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", gray_frame)
        key = cv2.waitKey(1)
        if key == 1:
            break
    cap.release()
    cv2.destroyAllWindows()