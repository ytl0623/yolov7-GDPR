import cv2

image1 = cv2.imread("1_pose.jpg")
image1 = cv2.resize(image1, (256, 256))
image2 = cv2.imread("1.jpg")
image2 = cv2.resize(image2, (256, 256))
image3 = cv2.subtract(image1, image2)

cv2.imshow("123",image3)
cv2.waitKey(0)
