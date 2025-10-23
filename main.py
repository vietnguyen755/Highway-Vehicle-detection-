"""
Main Application for Vietnamese Traffic Vehicle Detection and Counting
This script performs real-time vehicle detection, tracking, and counting.
"""

import cv2
import numpy as np
import os
import glob
import argparse
from collections import defaultdict
from ultralytics import YOLO
import math

class CentroidTracker:
    """
    Simple centroid tracking algorithm to track objects across frames.
    """
    
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        Initialize the centroid tracker.
        
        Args:
            max_disappeared (int): Maximum frames an object can be missing before removal
            max_distance (int): Maximum distance for object association
        """
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store object centroids
        self.disappeared = {}  # Dictionary to track disappeared frames
        self.object_classes = {}  # Dictionary to store object classes
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """
        Register a new object with the given centroid.
        
        Args:
            centroid (tuple): (x, y) coordinates of the centroid
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """
        Deregister an object by removing it from tracking.
        
        Args:
            object_id (int): ID of the object to remove
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.object_classes:
            del self.object_classes[object_id]
    
    def update(self, rects):
        """
        Update the tracker with new detections.
        
        Args:
            rects (list): List of bounding boxes [(x1, y1, x2, y2), ...]
        
        Returns:
            dict: Dictionary mapping object_id to centroid
        """
        # If no detections, mark all objects as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Calculate centroids for new detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        # If no existing objects, register all new detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match existing objects with new detections
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = list(self.objects.keys())[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched objects and detections
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # If more objects than detections, mark unmatched objects as disappeared
            if D.shape[0] >= D.shape[1]:
                object_ids = list(self.objects.keys())
                for row in unused_row_indices:
                    if row < len(object_ids):
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
            else:
                # More detections than objects, register new objects
                for col in unused_col_indices:
                    self.register(input_centroids[col])
        
        return self.objects

class VehicleCounter:
    """
    Main class for vehicle detection, tracking, and counting.
    """
    
    def __init__(self, model_path="runs/detect/yolov8m_stage2_improved/weights/best.pt", video_path=None, output_path=None):
        """
        Initialize the vehicle counter.
        
        Args:
            model_path (str): Path to the trained YOLO model
            video_path (str): Path to video file (optional, will auto-detect if None)
            output_path (str): Path to save output video (None for live display)
        """
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.model = None
        self.tracker = CentroidTracker()
        
        # Vehicle class mapping (adjust based on your training data)
        self.class_names = {
            0: 'auto',
            1: 'bus', 
            2: 'car',
            3: 'lcv',
            4: 'motorcycle',
            5: 'multiaxle',
            6: 'tractor',
            7: 'truck'
        }
        
        # Classification correction mapping (fix misclassifications)
        self.classification_corrections = {
            'motorcycle': 'truck'  # Fix: motorcycles are actually trucks
        }
        
        # Counters for each vehicle type
        self.counts = {'auto': 0, 'bus': 0, 'car': 0, 'lcv': 0, 'motorcycle': 0, 'multiaxle': 0, 'tractor': 0, 'truck': 0}
        
        # Track which objects have been counted (to avoid double counting)
        self.counted_objects = set()
        
        # Counting line position (horizontal line across the frame)
        self.counting_line_y = None
        
        # Colors for different vehicle types
        self.colors = {
            'auto': (0, 255, 0),        # Green
            'bus': (255, 0, 0),         # Blue
            'car': (0, 0, 255),         # Red
            'lcv': (255, 255, 0),       # Cyan
            'motorcycle': (255, 0, 255), # Magenta
            'multiaxle': (0, 255, 255),  # Yellow
            'tractor': (128, 0, 128),    # Purple
            'truck': (255, 165, 0)       # Orange
        }
    
    def correct_classification(self, vehicle_type):
        """
        Apply classification corrections to fix misclassifications.
        
        Args:
            vehicle_type (str): Original vehicle type
            
        Returns:
            str: Corrected vehicle type
        """
        return self.classification_corrections.get(vehicle_type, vehicle_type)
    
    def load_model(self):
        """
        Load the trained YOLO model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                print(f"Loading trained model from: {self.model_path}")
                self.model = YOLO(self.model_path)
                print("✓ Model loaded successfully!")
                return True
            else:
                print(f"✗ Trained model not found at: {self.model_path}")
                print("Please run train.py first to train the model.")
                return False
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return False
    
    def find_video_file(self):
        """
        Find the video file in the project directory.
        
        Returns:
            str: Path to video file, or None if not found
        """
        # If video path is provided, use it
        if self.video_path:
            if os.path.exists(self.video_path):
                return self.video_path
            else:
                print(f"✗ Video file not found: {self.video_path}")
                return None
        
        # Auto-detect first .mp4 file in directory
        mp4_files = glob.glob("*.mp4")
        if mp4_files:
            return mp4_files[0]
        return None
    
    def setup_counting_line(self, frame_height, frame_width):
        """
        Setup the counting line position.
        
        Args:
            frame_height (int): Height of the video frame
            frame_width (int): Width of the video frame
        """
        # Set counting line at 60% of frame height (adjust as needed)
        self.counting_line_y = int(frame_height * 0.6)
        print(f"Counting line set at y = {self.counting_line_y}")
    
    def has_crossed_line(self, object_id, centroid):
        """
        Check if an object has crossed the counting line.
        
        Args:
            object_id (int): ID of the tracked object
            centroid (tuple): Current centroid position (x, y)
        
        Returns:
            bool: True if object crossed the line, False otherwise
        """
        # Simple line crossing detection
        # You can enhance this with direction detection if needed
        if object_id not in self.counted_objects:
            if centroid[1] >= self.counting_line_y:
                self.counted_objects.add(object_id)
                return True
        return False
    
    def draw_overlay(self, frame, objects, detections):
        """
        Draw bounding boxes, tracking IDs, counting line, and count overlay.
        
        Args:
            frame: OpenCV frame
            objects (dict): Dictionary of tracked objects
            detections: YOLO detection results
        """
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), (frame.shape[1], self.counting_line_y), (255, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (10, self.counting_line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw bounding boxes and labels for detections
        if detections is not None and len(detections) > 0:
            try:
                for detection in detections:
                    if detection.boxes is not None:
                        boxes = detection.boxes.xyxy.cpu().numpy()
                        confidences = detection.boxes.conf.cpu().numpy()
                        class_ids = detection.boxes.cls.cpu().numpy()
                        
                        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                            if conf > 0.5:  # Confidence threshold
                                x1, y1, x2, y2 = map(int, box)
                                class_name = self.class_names.get(int(class_id), 'unknown')
                                # Apply classification correction
                                corrected_name = self.correct_classification(class_name)
                                color = self.colors.get(corrected_name, (128, 128, 128))
                                
                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Draw label
                                label = f"{corrected_name}: {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"Warning: Error in draw_overlay: {e}")
                pass
        
        # Draw tracking IDs
        for object_id, centroid in objects.items():
            cv2.circle(frame, centroid, 5, (0, 255, 255), -1)
            cv2.putText(frame, f"ID: {object_id}", (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw count overlay
        y_offset = 30
        for vehicle_type, count in self.counts.items():
            color = self.colors.get(vehicle_type, (128, 128, 128))
            text = f"{vehicle_type.capitalize()}: {count}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30
    
    def process_video(self, video_path):
        """
        Process the video file for vehicle detection and counting.
        
        Args:
            video_path (str): Path to the video file
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f} seconds")
        
        # Setup counting line
        self.setup_counting_line(frame_height, frame_width)
        
        # Setup video writer if output path is specified
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
            print(f"Output video will be saved to: {self.output_path}")
        
        print("\nStarting video processing...")
        if not self.output_path:
            print("Press 'q' to quit, 'p' to pause")
        
        frame_count = 0
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                
                # Extract bounding boxes
                boxes = []
                vehicle_classes = []
                
                if results[0].boxes is not None:
                    detection_boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy()
                    
                    for box, conf, class_id in zip(detection_boxes, confidences, class_ids):
                        if conf > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = map(int, box)
                            boxes.append((x1, y1, x2, y2))
                            vehicle_classes.append(int(class_id))
                
                # Update tracker
                tracked_objects = self.tracker.update(boxes)
                
                # Store vehicle classes for tracked objects
                for i, (object_id, centroid) in enumerate(tracked_objects.items()):
                    if i < len(vehicle_classes):
                        self.tracker.object_classes[object_id] = vehicle_classes[i]
                
                # Check for line crossings and update counts
                for object_id, centroid in tracked_objects.items():
                    if self.has_crossed_line(object_id, centroid):
                        # Get the stored vehicle class for this object
                        vehicle_class = self.tracker.object_classes.get(object_id, 0)
                        vehicle_type = self.class_names.get(vehicle_class, 'unknown')
                        # Apply classification correction
                        corrected_type = self.correct_classification(vehicle_type)
                        if corrected_type in self.counts:
                            self.counts[corrected_type] += 1
                            print(f"Vehicle counted: {corrected_type} (ID: {object_id})")
                
                # Draw overlay
                try:
                    self.draw_overlay(frame, tracked_objects, results)
                except Exception as e:
                    print(f"Warning: Error in draw_overlay: {e}")
                    pass
                
                # Write frame to output video if specified
                if out is not None:
                    out.write(frame)
                
                # Display frame only if no output video is being saved
                if self.output_path is None:
                    try:
                        cv2.imshow('Highway Traffic Vehicle Detection', frame)
                        
                        # Handle key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("Quit requested by user")
                            break
                        elif key == ord('p'):
                            print("Paused. Press any key to continue...")
                            cv2.waitKey(0)
                    except cv2.error as e:
                        print(f"Display error: {e}")
                        print("Continuing without display...")
                else:
                    # For video output, just check for quit
                    if frame_count % 1000 == 0:  # Check every 1000 frames
                        print(f"Processing frame {frame_count}...")
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                break
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
            print(f"✓ Output video saved to: {self.output_path}")
        
        if self.output_path is None:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    
    def save_results(self):
        """
        Save the counting results to a text file.
        """
        results_file = "results.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Vietnamese Traffic Vehicle Detection Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("Vehicle Count Summary:\n")
            f.write("-" * 25 + "\n")
            
            total_vehicles = 0
            for vehicle_type, count in self.counts.items():
                f.write(f"{vehicle_type.capitalize()}: {count}\n")
                total_vehicles += count
            
            f.write("-" * 25 + "\n")
            f.write(f"Total Vehicles: {total_vehicles}\n")
            f.write("\nDetection completed successfully!\n")
        
        print(f"\n✓ Results saved to: {results_file}")
        print("\nFinal Count Summary:")
        print("-" * 25)
        for vehicle_type, count in self.counts.items():
            print(f"{vehicle_type.capitalize()}: {count}")
        print("-" * 25)
        print(f"Total Vehicles: {sum(self.counts.values())}")

def main():
    """
    Main function to run the vehicle detection and counting application.
    """
    parser = argparse.ArgumentParser(description='Vietnamese Traffic Vehicle Detection & Counting')
    parser.add_argument('--model', default='runs/detect/train/weights/best.pt', 
                       help='Path to trained YOLO model (default: runs/detect/train/weights/best.pt)')
    parser.add_argument('--video', default=None,
                       help='Path to video file to process (default: auto-detect first .mp4 in directory)')
    parser.add_argument('--output', default=None,
                       help='Path to save output video (default: None, display live)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vietnamese Traffic Vehicle Detection & Counting")
    print("=" * 60)
    print(f"Model: {args.model}")
    
    # Initialize vehicle counter with specified model and video
    counter = VehicleCounter(model_path=args.model, video_path=args.video, output_path=args.output)
    
    # Load the trained model
    if not counter.load_model():
        return 1
    
    # Find video file
    video_path = counter.find_video_file()
    if not video_path:
        print("✗ No .mp4 video file found in the current directory!")
        return 1
    
    print(f"✓ Found video file: {video_path}")
    
    # Process the video
    try:
        counter.process_video(video_path)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        return 1
    
    # Save results
    counter.save_results()
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
