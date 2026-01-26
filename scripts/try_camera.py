import cv2

# Use 0 for the default MacBook webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Macbook Camera', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
