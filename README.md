# Neural Style Transfer with Python
Apply pre-trained model style on to an input image to blend two image so that the output would look like the original image but with the style of the reference model.

<img src="./Hau's_Picture.jpg" alt="Hau" width="400"/> <img src="output\Hau's_Picture_feathers.jpg" alt="transferred" width="400"/>


<img src="dog.jpg" alt="Hau" width="400"/> <img src="output\dog_mosaic.jpg" alt="transferred" width="400"/>

## Usage
- Single image: `python neural_style_transfer_single_image.py -m models/eccv16/starry_night.t7 -i image_path`
- All models for a single image: `python neural_style_transfer_all_models.py -m models -i image_path`
- Live webcam (slow): `python neural_style_transfer_video.py -m models`

## Dependencies
- OpenCV2
- imutils

