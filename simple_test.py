import sys, os
import cv2


classifier_model = '/opt/opencv/data/lbpcascades/lbpcascade_frontalface.xml'

if __name__=='__main__':
    image_path  = sys.argv[1]
    assert(os.path.exists(image_path)), 'Image not found'

    # careate cascade classifier object
    cascade = cv2.CascadeClassifier(classifier_model)

    # read the image
    image = cv2.imread(image_path)
    assert(image != None), 'Unable to read the image'

    # convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = cascade.detectMultiScale(gray_img,
        scaleFactor = 1.1,
        minNeighbors = 3,
        minSize = (30, 30),
        maxSize = (gray_img.shape[1], gray_img.shape[0])
    )
    print faces
    print ('No of faces detected = {:d}'.format(len(faces)))
        
    # draw rectange over the image and show the image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)

    cv2.namedWindow('detected_faces')
    cv2.imshow('detected_faces', image)
    cv2.waitKey()
