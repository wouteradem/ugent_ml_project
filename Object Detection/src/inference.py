import numpy as np
import cv2
import torch
import glob as glob
from model import create_model

# set the computation device
device = torch.device('cpu')

# load the model and the trained weights
model = create_model(num_classes=2).to(device)
model.load_state_dict(torch.load('../output/model70.pth'))
model.eval()

# directory where all the images are present
DIR_TEST = '../test_data'

test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")
# classes: 0 index is reserved for background
CLASSES = ['background', 'Signature']

detection_threshold = 0.8

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    print(image_name)
    print(test_images[i])
    image = cv2.imread(test_images[i])
    orig_image = image.copy()

    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float64)

    # make the pixel range between 0 and 1
    image /= 255.0

    # bring color channels to front
    image = np.transpose(image, (2, 0, 1))

    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)

    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j],
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2, lineType=cv2.LINE_AA)

        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
        cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image)
    print(f"Image {i + 1} done...")
