## Requirements
- Ensure you have Python version **3.11.6** or higher installed.
- Install the **ultralytics** package:
  ```bash
  pip install ultralytics -U
  ```

## Repository Unpacking
- Download the repository (click the green "Code" button and then "Download ZIP"), then extract the archive to a convenient location so that everything is in one folder.

## Annotating Images / Creating a Dataset / Tips
- There are many programs and services for annotating images. If you're doing this for the first time, it's recommended to use **LabelImg**.

### Installing and Setting up LabelImg
1. Download [LabelImg](https://github.com/HumanSignal/labelImg) or [LabelImg_Next](https://github.com/SunOner/labelImg_Next).
2. Extract the program and move the `predefined_classes.txt` file (located in the repository) to the `labelimg/data` folder.

### Repository Structure with Annotated Image Examples
The repository already contains examples of annotated images in the `ai_aimbot_train/datasets/game` folder:
- `images` – contains images for model training.
- `labels` – contains annotations for the corresponding images. These are text files with the coordinates of bounding boxes and object class IDs.
- `val` – used to store images and annotations for the validation dataset. The validation set is needed to check the model's performance during training and prevent overfitting.

### Annotating Your Own Data
1. Open LabelImg and click **Open Dir**, selecting the path to the images for model training (`ai_aimbot_train/datasets/game/images`).
2. Click **Change Save Dir** to choose where the annotations will be saved (`ai_aimbot_train/datasets/game/labels`).
3. (Optional) Familiarize yourself with the [LabelImg usage guide](https://github.com/HumanSignal/labelImg?tab=readme-ov-file#steps-yolo).
4. (Optional) Explore [tips for achieving the best training results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/).

### Recommendations for Creating a Quality Dataset
- Annotate between **500 to 2500** images.
- Include empty images that lack players, weapons, fire, and other objects. For example, add lots of images with trees, chairs, grass, human-like objects, and empty game locations.
- The more complex a game looks for AI (e.g., **CS2** is more formalized than **Battlefield 2042**), the more data you'll need for model training (at least **5000-10000** images).
- Image resolution can vary from **100x100** to **4K**.
- Ready-made datasets can be found [here](https://universe.roboflow.com/).
- Don't forget to add images and annotated files to the `val` folder. If you have **1000** training images, add about **10** validation images.

### Setting up the `game.yaml` File
- The file specifies the following parameters:
  ```yaml
  path: game  # Your dataset name
  train: images  # Folder with training images
  val: val  # Folder with validation images
  test:  # Folder with test images (can be empty)
  ```

- To find Ultralytics settings file, enter the command:
  ```bash
  yolo settings
  ```

## Model Training
- (Optional) Detailed information about training can be found [here](https://docs.ultralytics.com/modes/train/).
- After annotating the dataset, navigate to the `ai_aimbot_train` folder using the command:
  ```bash
  cd ai_aimbot_train
  ```
- Choose a pre-trained model. The options are:
  - **yolo11n** or **yolo12n** – the fastest and least resource-intensive.
  - **yolo11s** or **yolo12s** – fast, slightly smarter but more resource-demanding.
  - **yolo11m** or **yolo12m** – optimized for real-time, requires a powerful GPU (e.g., RTX 2060 or better).
  - **yolo11l** or **yolo12l** and **yolo11x** – most intelligent and resource-intensive, not suitable for most tasks.

- For example, choose **yolo11n**:
  ```bash
  model = yolo11n.pt
  ```

- Select the image size for the model. The lower the resolution, the faster the training and the fewer objects the model can detect:
  - Possible options: **160**, **320**, **480**, **640**. Choose **320**.

- Determine the number of training epochs. A larger dataset requires more epochs, but don't overdo it, as too many epochs can lead to overfitting. Assume we set **40** epochs.

- (Optional) All training parameters are described in detail [here](https://docs.ultralytics.com/modes/train/#train-settings).

- Start training with the command:
  ```bash
  yolo detect train data=game.yaml model=yolo11n.pt epochs=40 batch=-1
  ```

- After successful training, navigate to the model weights folder:
  ```
  ai_aimbot_train/runs/detect/train/weights
  ```
  You'll see two files:
  - `best.pt` – the file with the best model weights.
  - `last.pt` – checkpoint of the last epoch. If training was interrupted, you can resume from this file:
    ```bash
    yolo train resume model=ai_aimbot_train/runs/detect/train/weights/last.pt
    ```

## Model Testing
- Create a `test.py` file with the following content:
  ```python
	import cv2

	from ultralytics import YOLO
	
	# Load the YOLO model
	model = YOLO("best.pt")
	
	# Open the video file
	video_path = "path/to/your/video/file.mp4"
	cap = cv2.VideoCapture(video_path)
	
	# Loop through the video frames
	while cap.isOpened():
		# Read a frame from the video
		success, frame = cap.read()
	
		if success:
			# Run YOLO inference on the frame
			results = model(frame)
	
			# Visualize the results on the frame
			annotated_frame = results[0].plot()
	
			# Display the annotated frame
			cv2.imshow("YOLO Inference", annotated_frame)
	
			# Break the loop if 'q' is pressed
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
		else:
			# Break the loop if the end of the video is reached
			break
	
	# Release the video capture object and close the display window
	cap.release()
	cv2.destroyAllWindows()
  ```
- Run:
  ```bash
  python test.py
  ```
  
## Fine-Tuning the Sunxds Model to Avoid False Detections or Improve Detection of Other Classes

1. **Selecting a Pre-trained Sunxds Model**
   - Choose a model that is most similar to your task in terms of version and image size (640, 480, 320 or 160).

2. **Preparing the Dataset**
   - Prepare the dataset with new annotations as specified in this repository. Ensure that the new annotations follow the YOLO format.
   - If you want to eliminate false detections, add images with false detections to the dataset and do not annotate anything on them.
   - If, for some reason, there are poor or no detections of players in a particular game, add annotated images with these players to the dataset.
   - Don't be lazy; add more images to the dataset, and try to review the entire dataset again for errors after annotating it.

3. **Running the Fine-Tuning**
   - Execute the following command:
     ```bash
     yolo detect train data=game.yaml model=sunxds_0.7.5.pt epochs=40
     ```

## Export model to run model in sunone_aimbot_cpp:
- Export model to onnx format with dynamic shape:
	```bash
	yolo export model=best.pt format=onnx dynamic=true simplify=true
	```
- Select .onnx model in AI tab.