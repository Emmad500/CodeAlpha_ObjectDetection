import cv2
from ultralytics import YOLO

# --- 1. Load the AI Model ---
# We use 'yolov8n.pt' which is the standard Nano model for speed.
print("Loading AI Brain... Please wait.")
model = YOLO('yolov8n.pt') 

# --- 2. Initialize Video Capture ---
cap = cv2.VideoCapture('test_video.mp4')
cap.set(3, 1280) # Set Width
cap.set(4, 720)  # Set Height

if not cap.isOpened():
    print("Error: Webcam not found.")
    exit()

print("System Active. Press 'q' to exit.")

while True:
    success, frame = cap.read()
    
    if success:
        # --- 3. Run Tracking (Requirement: Object Tracking & IDs) ---
        # persist=True tells the AI to remember objects between frames (Tracking)
        results = model.track(frame, persist=True)

        # --- 4. Better UI: Draw the AI Vision ---
        # This automatically draws the boxes AND the Tracking IDs (e.g., "id: 1")
        annotated_frame = results[0].plot()

        # --- 5. Add a "Dashboard" (UI Improvement) ---
        # We overlay a rectangle and text to make it look like a sci-fi interface
        height, width, _ = frame.shape
        
        # Draw a semi-transparent black bar at the top
        cv2.rectangle(annotated_frame, (0, 0), (width, 50), (0, 0, 0), -1)
        
        # Add the Title Text
        cv2.putText(annotated_frame, "CODEALPHA AI VISION SYSTEM", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add 'Press Q' instruction
        cv2.putText(annotated_frame, "Press 'Q' to Quit", (width - 250, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- 6. Display Output ---
        cv2.imshow("CodeAlpha Task 4: Object Tracking", annotated_frame)

        # --- 7. Exit Logic ---
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()