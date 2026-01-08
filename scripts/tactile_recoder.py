#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from xela_server_ros.msg import SensStream
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import json
import numpy as np
from datetime import datetime
import threading


class MultiModalRecorder:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('multimodal_recorder', anonymous=True)
        
        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/camera/raw_image')
        self.xela_topic = rospy.get_param('~xela_topic', 'xServTopic')
        self.save_dir = rospy.get_param('~save_dir', './recorded_data')
        self.image_format = rospy.get_param('~image_format', 'jpg')  # jpg or png
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # CvBridge
        self.bridge = CvBridge()
        
        # Recording state
        self.recording = False
        self.image_frame_count = 0
        self.xela_frame_count = 0
        self.session_count = 0
        self.current_session_dir = None
        self.current_image_dir = None
        self.current_xela_file = None
        
        # XELA data accumulator for the current session
        self.xela_data_list = []
        self.xela_lock = threading.Lock()
        
        # Latest data
        self.latest_image = None
        self.latest_xela = None
        self.image_lock = threading.Lock()
        
        # Subscribe to topics
        self.image_subscriber = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.xela_subscriber = rospy.Subscriber(self.xela_topic, SensStream, self.xela_callback)
        
        rospy.loginfo("="*60)
        rospy.loginfo("Multi-Modal Recorder Started")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Image topic: {self.image_topic}")
        rospy.loginfo(f"XELA topic: {self.xela_topic}")
        rospy.loginfo(f"Save directory: {self.save_dir}")
        rospy.loginfo(f"Image format: {self.image_format}")
        rospy.loginfo(f"XELA format: Single .npy file per session")
        rospy.loginfo("")
        rospy.loginfo("KEYBOARD CONTROLS:")
        rospy.loginfo("  's' - Start recording (both image & XELA)")
        rospy.loginfo("  'e' - End/Stop recording")
        rospy.loginfo("  'p' - Pause/Resume recording")
        rospy.loginfo("  'c' - Capture single frame (image & XELA snapshot)")
        rospy.loginfo("  'q' - Quit")
        rospy.loginfo("="*60)
        rospy.loginfo("")
        
    def image_callback(self, msg):
        """Callback function for image topic"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.image_lock:
                self.latest_image = cv_image
            
            # Save frame if recording
            if self.recording and self.current_image_dir is not None:
                self.save_image_frame(cv_image)
                
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")
    
    def xela_callback(self, msg):
        """Callback function for XELA sensor topic"""
        try:
            with self.xela_lock:
                self.latest_xela = msg
            
            # Accumulate data if recording
            if self.recording:
                self.accumulate_xela_data(msg)
                
        except Exception as e:
            rospy.logerr(f"Error in XELA callback: {e}")
    
    def save_image_frame(self, frame):
        """Save an image frame to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(
            self.current_image_dir,
            f"frame_{self.image_frame_count:06d}_{timestamp}.{self.image_format}"
        )
        
        try:
            cv2.imwrite(filename, frame)
            self.image_frame_count += 1
            
            # Log every 30 frames
            if self.image_frame_count % 30 == 0:
                rospy.loginfo(f"📷 Images: {self.image_frame_count} | 🖐️ XELA: {self.xela_frame_count}")
        except Exception as e:
            rospy.logerr(f"Error saving image frame: {e}")
    
    def accumulate_xela_data(self, msg):
        """Accumulate XELA sensor data into a list"""
        try:
            timestamp = datetime.now().timestamp()
            
            # Convert XELA message to structured data
            frame_data = {
                'timestamp': timestamp,
                'frame_number': self.xela_frame_count,
                'sensors': []
            }
            
            for sensor in msg.sensors:
                sensor_data = {
                    'sensor_pos': sensor.sensor_pos,
                    'model': sensor.model,
                    'message': sensor.message,
                    'time': sensor.time,
                    'taxels': [],
                    'forces': []
                }
                
                # Add taxel data
                for taxel in sensor.taxels:
                    sensor_data['taxels'].append([taxel.x, taxel.y, taxel.z])
                
                # Add force data
                for force in sensor.forces:
                    sensor_data['forces'].append([force.x, force.y, force.z])
                
                frame_data['sensors'].append(sensor_data)
            
            with self.xela_lock:
                self.xela_data_list.append(frame_data)
                self.xela_frame_count += 1
            
        except Exception as e:
            rospy.logerr(f"Error accumulating XELA data: {e}")
    
    def save_xela_to_npy(self):
        """Save all accumulated XELA data to a single .npy file"""
        if not self.xela_data_list:
            rospy.logwarn("No XELA data to save")
            return
        
        try:
            # Save as .npy file
            npy_filename = os.path.join(self.current_session_dir, 'xela_data.npy')
            np.save(npy_filename, self.xela_data_list, allow_pickle=True)
            
            # Also save a JSON version for easy inspection
            json_filename = os.path.join(self.current_session_dir, 'xela_data.json')
            with open(json_filename, 'w') as f:
                json.dump(self.xela_data_list, f, indent=2)
            
            rospy.loginfo(f"✅ XELA data saved to: {npy_filename}")
            rospy.loginfo(f"✅ XELA JSON saved to: {json_filename}")
            
        except Exception as e:
            rospy.logerr(f"Error saving XELA data: {e}")

    def get_next_episode_number(self):
        """Find the next available episode number"""
        if not os.path.exists(self.save_dir):
            return 1
        
        # Find all existing episode directories
        existing_episodes = []
        for item in os.listdir(self.save_dir):
            if item.startswith('episode_') and os.path.isdir(os.path.join(self.save_dir, item)):
                try:
                    # Extract episode number
                    episode_num = int(item.split('_')[1])
                    existing_episodes.append(episode_num)
                except (IndexError, ValueError):
                    continue
        
        # Return next episode number
        if existing_episodes:
            return max(existing_episodes) + 1
        else:
            return 1
    
    def start_recording(self):
        """Start a new recording session"""
        if self.recording:
            rospy.logwarn("Already recording!")
            return
        
        self.session_count += 1
        episode_num = self.get_next_episode_number()
        session_name = f"episode_{episode_num}"
        self.current_session_dir = os.path.join(self.save_dir, session_name)
        
        # Create subdirectory for images
        self.current_image_dir = os.path.join(self.current_session_dir, 'images')
        os.makedirs(self.current_image_dir, exist_ok=True)
        
        # Create main session directory (XELA will be saved here as single file)
        os.makedirs(self.current_session_dir, exist_ok=True)
        
        # Create session metadata file
        metadata_file = os.path.join(self.current_session_dir, 'session_info.json')
        metadata = {
            'session_id': self.session_count,
            'start_time': datetime.now().isoformat(),
            'image_topic': self.image_topic,
            'xela_topic': self.xela_topic,
            'image_format': self.image_format,
            'xela_format': 'npy'
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Reset counters and data
        self.image_frame_count = 0
        self.xela_frame_count = 0
        with self.xela_lock:
            self.xela_data_list = []
        
        self.recording = True
        
        rospy.loginfo("")
        rospy.loginfo("="*60)
        rospy.loginfo(f"🔴 RECORDING STARTED - Session {self.session_count}")
        rospy.loginfo(f"Session dir: {self.current_session_dir}")
        rospy.loginfo(f"  Images -> {self.current_image_dir}/")
        rospy.loginfo(f"  XELA   -> {self.current_session_dir}/xela_data.npy")
        rospy.loginfo("="*60)
        rospy.loginfo("")
    
    def stop_recording(self):
        """Stop the current recording session"""
        if not self.recording:
            rospy.logwarn("Not currently recording!")
            return
        
        self.recording = False
        
        # Save all accumulated XELA data to .npy file
        self.save_xela_to_npy()
        
        # Update metadata with end time and frame counts
        metadata_file = os.path.join(self.current_session_dir, 'session_info.json')
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['end_time'] = datetime.now().isoformat()
            metadata['total_image_frames'] = self.image_frame_count
            metadata['total_xela_frames'] = self.xela_frame_count
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            rospy.logerr(f"Error updating metadata: {e}")
        
        rospy.loginfo("")
        rospy.loginfo("="*60)
        rospy.loginfo(f"⏹ RECORDING STOPPED")
        rospy.loginfo(f"Image frames saved: {self.image_frame_count}")
        rospy.loginfo(f"XELA frames saved: {self.xela_frame_count}")
        rospy.loginfo(f"Location: {self.current_session_dir}")
        rospy.loginfo("="*60)
        rospy.loginfo("")
        
        self.current_session_dir = None
        self.current_image_dir = None
    
    def pause_recording(self):
        """Toggle pause/resume recording"""
        if not self.current_session_dir:
            rospy.logwarn("No active session to pause!")
            return
        
        self.recording = not self.recording
        
        if self.recording:
            rospy.loginfo("▶️  RECORDING RESUMED")
        else:
            rospy.loginfo("⏸  RECORDING PAUSED")
    
    def capture_single_frame(self):
        """Capture a single snapshot of both image and XELA data"""
        with self.image_lock:
            image_available = self.latest_image is not None
            if image_available:
                image_frame = self.latest_image.copy()
        
        with self.xela_lock:
            xela_available = self.latest_xela is not None
            if xela_available:
                xela_data = self.latest_xela
        
        if not image_available and not xela_available:
            rospy.logwarn("No data available to capture")
            return
        
        # Create single_captures directory with subdirectories
        single_dir = os.path.join(self.save_dir, 'single_captures')
        single_image_dir = os.path.join(single_dir, 'images')
        single_xela_dir = os.path.join(single_dir, 'xela')
        
        os.makedirs(single_image_dir, exist_ok=True)
        os.makedirs(single_xela_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save image
        if image_available:
            image_filename = os.path.join(single_image_dir, f"capture_{timestamp}.{self.image_format}")
            try:
                cv2.imwrite(image_filename, image_frame)
                rospy.loginfo(f"📸 Image captured: {image_filename}")
            except Exception as e:
                rospy.logerr(f"Error capturing image: {e}")
        
        # Save XELA data as .npy
        if xela_available:
            try:
                # Convert to structured data (single frame)
                data = {
                    'timestamp': timestamp,
                    'sensors': []
                }
                
                for sensor in xela_data.sensors:
                    sensor_data = {
                        'sensor_pos': sensor.sensor_pos,
                        'model': sensor.model,
                        'message': sensor.message,
                        'time': sensor.time,
                        'taxels': [[t.x, t.y, t.z] for t in sensor.taxels],
                        'forces': [[f.x, f.y, f.z] for f in sensor.forces]
                    }
                    data['sensors'].append(sensor_data)
                
                # Save as .npy
                npy_filename = os.path.join(single_xela_dir, f"capture_{timestamp}.npy")
                np.save(npy_filename, data, allow_pickle=True)
                
                # Also save JSON for inspection
                json_filename = os.path.join(single_xela_dir, f"capture_{timestamp}.json")
                with open(json_filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                rospy.loginfo(f"🖐️  XELA captured: {npy_filename}")
            except Exception as e:
                rospy.logerr(f"Error capturing XELA data: {e}")
    
    def show_preview(self):
        """Show live preview window with recording status"""
        window_name = "Multi-Modal Recorder - Preview"
        
        while not rospy.is_shutdown():
            with self.image_lock:
                if self.latest_image is not None:
                    frame = self.latest_image.copy()
                else:
                    continue
            
            # Check XELA status
            with self.xela_lock:
                xela_available = self.latest_xela is not None
                if xela_available:
                    num_sensors = len(self.latest_xela.sensors)
                else:
                    num_sensors = 0
            
            # Add status overlay
            display_frame = frame.copy()
            height, width = display_frame.shape[:2]
            
            # Recording status
            if self.recording:
                status = "RECORDING"
                color = (0, 0, 255)  # Red
                cv2.circle(display_frame, (30, 30), 10, color, -1)
            else:
                status = "STANDBY"
                color = (0, 255, 0)  # Green
            
            cv2.putText(display_frame, status, (50, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Frame counts
            if self.recording:
                cv2.putText(display_frame, f"Images: {self.image_frame_count}", (50, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"XELA: {self.xela_frame_count}", (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # XELA sensor status
            xela_status = f"XELA: {num_sensors} sensors" if xela_available else "XELA: No data"
            xela_color = (0, 255, 255) if xela_available else (0, 0, 255)
            cv2.putText(display_frame, xela_status, (width - 250, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, xela_color, 2)
            
            # Show frame
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.start_recording()
            elif key == ord('e'):
                self.stop_recording()
            elif key == ord('p'):
                self.pause_recording()
            elif key == ord('c'):
                self.capture_single_frame()
            elif key == ord('q'):
                rospy.loginfo("Quit command received")
                break
        
        cv2.destroyAllWindows()
    
    def run(self):
        """Main run loop"""
        # Start preview in main thread (OpenCV requires main thread)
        try:
            self.show_preview()
        except KeyboardInterrupt:
            rospy.loginfo("Interrupted by user")
        finally:
            if self.recording:
                self.stop_recording()
            rospy.loginfo("Shutting down...")


def main():
    try:
        recorder = MultiModalRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()