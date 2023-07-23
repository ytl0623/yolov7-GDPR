import cv2

vidcap = cv2.VideoCapture('4_mask.mp4')
vidcap2 = cv2.VideoCapture('4_backbone.mp4')

resize_width = 640
resize_height = 360

out = cv2.VideoWriter("4_coverVideo.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (resize_width, resize_height))

success1, image1 = vidcap.read()
success2, image2 = vidcap2.read()

while success1 :
  image1 = cv2.resize(image1, (640, 360))
  image2 = cv2.resize(image2, (640, 360))
  image3 = cv2.add(image1, image2)

  out.write(image3)
    
  success1, image1 = vidcap.read()
  success2, image2 = vidcap2.read()
  #print('Read a new frame: ', success1)
