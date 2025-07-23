import os
import cv2
import time
import uuid

IMAGE_PATH = 'CollectedImages'

labels = ['Hello', 'Yes', 'No', 'Thank You', 'I Love You', 'Please']

no_of_images = 20

for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)  # Allow camera to warm up
    for img_num in range(no_of_images):
        ret, frame = cap.read()
        if ret:
            img_name = os.path.join(img_path, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(img_name, frame)
            print('Image {} saved for label {}'.format(img_num + 1, label))
            time.sleep(2)  # Wait before capturing the next image
        else:
            print('Failed to capture image')
    cap.release()
print('Image collection completed.')
cv2.destroyAllWindows()
print('Images saved in {}'.format(IMAGE_PATH))
# End of image_processing.py
print('You can now proceed to train your model with the collected images.')