import cv2
import os
import csv
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize YOLOv8 object detector
yolov8_detector = YOLO("best.pt")

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')  # need to run only once to download and load model into memory

# Path to the folder containing images
image_folder = 'test'
output_txt_file = 'detected_plate_numbers.txt'
output_csv_file = 'detected_plate_numbers.csv'
output_image_folder = 'plate_recognitaion_images'

# Create the output image folder if it doesn't exist
os.makedirs(output_image_folder, exist_ok=True)

# Open the output files for writing
with open(output_txt_file, 'w') as txt_file, open(output_csv_file, 'w', newline='') as csv_file:
    txt_writer = txt_file
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image Name', 'Plate Number'])  # Writing header for CSV file

    # Iterate over each image in the folder
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):  # Adjust extensions as needed
            # Read image
            img_path = os.path.join(image_folder, image_name)
            img = cv2.imread(img_path)

            # Detect objects with specified thresholds
            results = yolov8_detector(img, conf=0.2, iou=0.3)

            # Extract boxes, scores, and class_ids from the first result
            boxes = results[0].boxes.xyxy.cpu().numpy()  # bounding box coordinates
            scores = results[0].boxes.conf.cpu().numpy()  # confidence scores
            class_ids = results[0].boxes.cls.cpu().numpy()  # class IDs

            # Define the class ID for plates (assuming it's 0, update if different)
            plate_class_id = 0

            # Iterate over the detected boxes
            for i, class_id in enumerate(class_ids):
                if class_id == plate_class_id:
                    # Extract the bounding box coordinates
                    x1, y1, x2, y2 = map(int, boxes[i])

                    # Crop the plate region from the image
                    plate_img = img[y1:y2, x1:x2]

                    # Perform OCR on the cropped plate image
                    result = ocr.ocr(plate_img, cls=False)

                    # Check if result is not None
                    if result:
                        # Extract and print plate numbers
                        for idx in range(len(result)):
                            res = result[idx]
                            if res:  # Check if res is not None
                                for line in res:
                                    text = line[1][0]
                                    print("Detected Plate Number:", text)
                                    # Write the result to the text file
                                    txt_writer.write(f"{image_name}: {text}\n")
                                    # Write the result to the CSV file
                                    csv_writer.writerow([image_name, text])
                                    # Draw bounding box and text on the original image
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                                    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)  # Draw text
                                    # Save the modified image
                                    cv2.imwrite(os.path.join(output_image_folder, image_name), img)
                                    break  # Assuming we only need one plate number per image
                    else:
                        print(f"No text detected in image: {image_name}")

print("Plate numbers and plate recognition images have been saved to", output_txt_file, ",", output_csv_file, "and", output_image_folder)
