import cv2
import numpy as np
import os

def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (B, G, R), -1)
    # return the pixelated blurred image
    return image

def load_and_mask_image(image):
    orig = image.copy()
    (h, w) = image.shape[:2]
    print("[INFO] loading face detector model...")
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    net = cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",
                                                caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is greater
        # than the minimum confidence
        if confidence > 0.7:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI
            face = image[startY:endY, startX:endX]

            face = anonymize_face_pixelate(face,
                    blocks=20)
            # store the blurred face in the output image
            image[startY:endY, startX:endX] = face

            # display the original image and the output image with the blurred
            # face(s) side by side
            # output = np.hstack([orig, image])
            # cv2.imshow("Output", output)
            # cv2.waitKey(0)
            #save the image
            cv2.imwrite(os.path.join("output" , "mask_" + filename), image)

folder = "images" # TODO: let's change it to read from input param

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        load_and_mask_image(img)






