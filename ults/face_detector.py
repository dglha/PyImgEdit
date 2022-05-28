import cv2
from PIL import Image
import numpy

CASC_PATH = "frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(CASC_PATH)

def face_detector(img: Image) -> Image:
    image = img.convert("L")
    image = numpy.array(image)

    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print(faces)

    img = numpy.array(img)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return Image.fromarray(img)