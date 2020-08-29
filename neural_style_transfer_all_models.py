from imutils import paths
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True, help="path to directory containing neural style transfer models")
ap.add_argument("-i", "--image", required=True, help="input image to apply neural style transfer to")
args = vars(ap.parse_args())

modelPaths = paths.list_files(args["models"], validExts=(".t7",))
modelPaths = sorted(list(modelPaths))

for modelPath in modelPaths:
	print("[INFO] loading {}...".format(modelPath))
	net = cv2.dnn.readNetFromTorch(modelPath)

	image = cv2.imread(args["image"])
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()

	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
	output = output.transpose(1, 2, 0)

	cv2.imshow("Input", image)
	cv2.imshow("Output", output)
	cv2.waitKey(0)

	if not os.path.isdir("output"):
		os.makedirs("output")
	output_name = f"output/{args['image'].split('.')[0]}_{os.path.basename(modelPath)[:-3]}.jpg"
	# NOTE: cv2.imshow() can handle range from [0, 1] but cv2.imwrite() needs a full color range to write properly
	cv2.imwrite(output_name, output*255.0)