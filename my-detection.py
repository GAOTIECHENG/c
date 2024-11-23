import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
display = jetson.utils.videoOutput("display://0") 



while display.IsStreaming():
	img = jetson.utils.loadImage("/home/nvidia/Desktop/cat.1.jpg")
	detections = net.Detect(img)
	for detection in detections:
		class_id = detection.ClassID
		confidence =detection.Confidence
		left = detection.Left
		top = detection.Top
		right = detection.Right
		bottom = detection.Bottom
		width = detection.Width
		height = detection.Height
		area = detection.Area
		center_x = detection.Center[0]
		center_y = detection.Center[1]
	print(f"Class ID: {class_id}")
	print(f"Confidence: {confidence}")
	print(f"Left: {left:.2f}, Top: {top:.2f}, Right: {right:.2f}, Bottom: {bottom:.2f}")
	print(f"Width: {width:.2f}, Height: {height:.2f}, Area: {area:.2f}")
	print(f"Center: ({center_x:.2f}, {center_y:.2f})")
	display.Render(img)
	display.SetStatus("Detecting objetcts...")
