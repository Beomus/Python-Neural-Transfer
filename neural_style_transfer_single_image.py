import argparse
import imutils
from imutils import paths
import cv2
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="neural style transfer model")
ap.add_argument("-i", "--image", required=True,
	help="input image to apply neural style transfer to")
args = vars(ap.parse_args())

# load the neural style model
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(args["model"])

# load and resize the input image
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image and then pass it forward through the network
blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
net.setInput(blob)
output = net.forward()

# reshape the output tensor, add back in the mean subtraction, then swap the channel ordering
output = output.reshape((3, output.shape[2], output.shape[3]))
output[0] += 103.939
output[1] += 116.779
output[2] += 123.680
output /= 255.0
output = output.transpose(1, 2, 0)

# show the images
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)

# save the image
if not os.path.isdir("output"):
	os.makedirs("output")
output_name = f"output/{args['image'].split('.')[0]}_{args['model'].split('/')[-1][:-3]}.jpg"
# NOTE: cv2.imshow() can handle range from [0, 1] but cv2.imwrite() needs a full color range to write properly
cv2.imwrite(output_name, output*255.0)
