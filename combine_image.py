import cv2

image1 = cv2.imread("1_instance.jpg")
image1 = cv2.resize(image1, (960, 704))
image2 = cv2.imread("1_backbone.jpg")
image2 = cv2.resize(image2, (960, 704))
image3 = cv2.add(image1, image2)

cv2.imwrite("coverImage.jpg",image3)
cv2.waitKey(0)
