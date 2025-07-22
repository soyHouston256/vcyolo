import cv2
import cvzone
import time
from ultralytics import YOLO
import argparse


def run_inference(model_name):
    # Loading the selected model
    if model_name == "yolo10":
        model = YOLO("yolov10n.pt")
    elif model_name == "yolo11":
        model = YOLO("yolo11n.pt")
    else:
        print("Invalid model name. Exiting.")
        return

    # Lists to store nums for respective measurements
    inferenceList = []
    fpsList = []
    confidenceList = []

    # Capturing live webcam footage
    cap = cv2.VideoCapture(0)

    # Inf loop to get all the frames
    while True:
        frameCaptured, frame = cap.read()

        if not frameCaptured:
            break

        # Getting time value of which the inferencing begins
        startTime = time.time()

        results = model(frame)

        # Plotting all objects found on frame without customising any of the visuals.
        processedFrame = results[0].plot()

        # Calculating inference time and converting to FPS
        inferenceTime = time.time() - startTime
        fps = 1 / inferenceTime

        # Adding values to a list
        inferenceList.append(inferenceTime)
        fpsList.append(fps)

        # Get confidence scores for all detected objects
        confidences = [box.conf[0] * 100 for box in results[0].boxes]
        if confidences:
            avgConfidence = sum(confidences) / len(confidences)
        else:
            # Handle case where no objects are detected
            avgConfidence = 0.0

        # Adding to a list
        confidenceList.append(avgConfidence)

        cvzone.putTextRect(processedFrame, f"FPS: {fps:.2f}", (10, 30), 2)
        cvzone.putTextRect(
            processedFrame, f"Inference Time: {inferenceTime:.4f} s", (10, 70), 2
        )

        cv2.imshow("YOLO Live Webcam", processedFrame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Getting average value of all the metrics
    avgFPS = sum(fpsList) / len(fpsList) if fpsList else 0
    avgInference = sum(inferenceList) / len(inferenceList) if inferenceList else 0
    avgConfidence = sum(confidenceList) / len(confidenceList) if confidenceList else 0

    # Printing values to console.
    print(f"Results for {model_name.upper()}:")
    print(f"Average FPS: {avgFPS:.2f}")
    print(f"Average Inference Time: {avgInference:.4f}")
    print(f"Average Confidence: {avgConfidence:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference on a webcam feed.")
    parser.add_argument('--model', type=str, default='yolo11', choices=['yolo10', 'yolo11'],
                        help='Choose the model to run: yolo10 or yolo11')
    args = parser.parse_args()
    run_inference(args.model)

# YOLO 10 results
# FPS: 13.04
# INFERENCE: 0.0796s
# CONF: 81.95%

# YOLO 11 RESULTS
# FPS: 19.10
# Inference: 0.0546s
# Conf: 70.05%