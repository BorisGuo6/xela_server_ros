#!/usr/bin/env python3
import rospy
import os
import json
import numpy as np
from datetime import datetime
import threading
import time

# --- Modules for Instant Keyboard Input ---
import sys
import tty
import termios
import select
# ----------------------------------------

# Import necessary ROS message types
from xela_server_ros.msg import SensStream
from std_msgs.msg import Float64

class KeyboardInputHandler:
    """Handles instant, non-blocking keyboard input."""
    def __init__(self, recorder):
        self.recorder = recorder
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        
        # Save original terminal settings
        self.settings = termios.tcgetattr(sys.stdin)

    def start(self):
        """Start the keyboard listener thread."""
        self.thread.start()

    def stop(self):
        """Stop the keyboard listener and restore terminal settings."""
        self.running = False
        self.thread.join()
        # Restore terminal settings
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        except termios.error as e:
            rospy.logwarn(f"Could not restore terminal settings: {e}. If the terminal looks strange, run 'reset'.")

    def _run(self):
        """Loop to read single characters from the terminal."""
        try:
            # Set the terminal to raw mode (non-canonical and no echo)
            tty.setraw(sys.stdin.fileno())
            rospy.loginfo("\n*** KEYBOARD LISTENER ACTIVE *** Press 's', 'e', 'p', or 'q' instantly.")
            
            # Use select to wait for input without blocking
            while self.running and not rospy.is_shutdown():
                # Check if input is available within 0.01 seconds
                if sys.stdin in select.select([sys.stdin], [], [], 0.01)[0]:
                    char = sys.stdin.read(1)
                    self._handle_command(char)
                # Small sleep to manage thread consumption
                time.sleep(0.01)
                
        except Exception as e:
            rospy.logerr(f"Keyboard handler error: {e}")
            self.running = False
            
    def _handle_command(self, command):
        """Execute the command based on the character received."""
        cmd = command.lower()
        
        if cmd == 's':
            self.recorder.start_recording()
        elif cmd == 'e':
            self.recorder.stop_recording()
        elif cmd == 'p':
            self.recorder.pause_recording()
        elif cmd == 'q':
            rospy.loginfo("Quit command received. Shutting down.")
            rospy.signal_shutdown("Quit command")
        elif cmd == '\x03': # Handle Ctrl+C (ASCII EOT)
            rospy.loginfo("Ctrl+C received. Shutting down.")
            rospy.signal_shutdown("User interrupt")


class ForceXelaRecorder:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('force_xela_recorder', anonymous=True)
        
        # Parameters
        self.ft_topic = rospy.get_param('~ft_topic', '/force_sensor/force')
        self.xela_topic = rospy.get_param('~xela_topic', 'xServTopic')
        self.save_dir = rospy.get_param('~save_dir', './recorded_force_xela_data')
        self.rate_hz = rospy.get_param('~rate_hz', 10.0)
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # State variables
        self.recording = False
        self.ft_frame_count = 0
        self.xela_frame_count = 0
        self.session_count = 0
        self.current_session_dir = None
        self.current_episode_num = 0
        
        # Data storage and locks
        self.ft_data_list = []
        self.xela_data_list = []
        self.lock = threading.Lock()
        
        # Latest received data (for synchronization)
        self.latest_ft_data = None
        self.latest_xela_msg = None
        
        # Subscribers
        self.ft_subscriber = rospy.Subscriber(self.ft_topic, Float64, self.ft_callback)
        self.xela_subscriber = rospy.Subscriber(self.xela_topic, SensStream, self.xela_callback)
        
        # ROS Rate for main recording loop
        self.rate = rospy.Rate(self.rate_hz)
        self.log_counter = 0

        # Initialize console output
        self._log_startup_info()

    def _log_startup_info(self):
        """Prints initial setup and control information to the console."""
        rospy.loginfo("="*60)
        rospy.loginfo("Force-Torque & XELA Recorder Started")
        rospy.loginfo("="*60)
        rospy.loginfo(f"FT topic: {self.ft_topic} (Type: std_msgs/Float64)")
        rospy.loginfo(f"XELA topic: {self.xela_topic}")
        rospy.loginfo(f"Recording Rate: {self.rate_hz} Hz")
        rospy.loginfo(f"Save directory: {self.save_dir}")
        rospy.loginfo("="*60)

    # --- Callback Functions ---
    
    def ft_callback(self, msg):
        with self.lock:
            self.latest_ft_data = msg.data
    
    def xela_callback(self, msg):
        with self.lock:
            self.latest_xela_msg = msg

    # --- Data Processing and Accumulation Helper ---

    def _convert_xela_to_72d_vector(self, msg):
        """Converts the SensStream message's force data into a 72D vector."""
        force_vector_72D = []
        
        if not msg.sensors:
            return None 

        for sensor in msg.sensors:
            # Check if forces array is present
            if not hasattr(sensor, 'forces') or not sensor.forces:
                 continue
                 
            # Flatten the x, y, z forces for each taxel
            for force in sensor.forces:
                force_vector_72D.extend([force.x, force.y, force.z])
                
        # Check if we got the expected 72 dimensions (24 taxels * 3 components)
        if len(force_vector_72D) == 72:
            return np.array(force_vector_72D)
        else:
            rospy.logwarn_throttle(2.0, f"XELA vector dimension mismatch: Expected 72, Got {len(force_vector_72D)}. Skipping frame.")
            return None

    def _record_and_log_frame(self):
        """Saves the latest data from both sensors and provides visualization logging."""
        
        with self.lock:
            ft_data = self.latest_ft_data
            xela_msg = self.latest_xela_msg
        
        if ft_data is None or xela_msg is None:
            rospy.logwarn_throttle(2.0, "Waiting for both Force and XELA data...")
            return
            
        xela_vector_72d = self._convert_xela_to_72d_vector(xela_msg)
        if xela_vector_72d is None:
            return

        # Calculate magnitudes for visualization/logging
        ft_magnitude = abs(ft_data)
        xela_magnitude = np.linalg.norm(xela_vector_72d)
            
        # 2. Accumulate Data (Synchronization happens here at 10 Hz rate)
        current_time = datetime.now().timestamp()
        
        # A. Force data accumulation
        ft_frame_data = {
            'frame_number': self.ft_frame_count,
            'timestamp': current_time,
            'force_value': ft_data
        }
        self.ft_data_list.append(ft_frame_data)
        self.ft_frame_count += 1
        
        # B. XELA data accumulation
        xela_frame_data = {
            'frame_number': self.xela_frame_count,
            'timestamp': current_time,
            'xela_force_72d': xela_vector_72d.tolist()
        }
        self.xela_data_list.append(xela_frame_data)
        self.xela_frame_count += 1
        
        # 3. Visualization/Logging (Every 10 frames)
        if self.ft_frame_count % 10 == 0:
            
            # --- ALIGNMENT FIX APPLIED HERE ---
            log_header = f"\n--- 🎬 EPISODE {self.current_episode_num:<3} | FRAMES {self.ft_frame_count:<6} ---"
            log_force = f"📊 Force (1D Magnitude): {ft_magnitude:>10.4f}" # Right-aligned float with 4 decimals, total width 10
            log_xela = f"🖐️ XELA (72D Norm):     {xela_magnitude:>10.4f}" # Added spaces and right-aligned float
            log_separator = "------------------------------------------"
            
            rospy.loginfo(log_header)
            rospy.loginfo(log_force)
            rospy.loginfo(log_xela)
            rospy.loginfo(log_separator)
            # ------------------------------------


    # --- Data Saving / Control Functions (Logic Unchanged) ---

    def _get_next_episode_number(self):
        """Find the next available episode number."""
        if not os.path.exists(self.save_dir): return 1
        existing_episodes = []
        for item in os.listdir(self.save_dir):
            if item.startswith('episode_') and os.path.isdir(os.path.join(self.save_dir, item)):
                try: existing_episodes.append(int(item.split('_')[1]))
                except (IndexError, ValueError): continue
        return max(existing_episodes) + 1 if existing_episodes else 1

    def save_data_to_npy(self):
        """Save all accumulated Force and XELA data to separate .npy files."""
        if not self.current_session_dir: return
            
        with self.lock:
            # 1. Save Force data
            if self.ft_data_list:
                ft_npy_filename = os.path.join(self.current_session_dir, 'force_data.npy')
                np.save(ft_npy_filename, self.ft_data_list, allow_pickle=True)
                rospy.loginfo(f"✅ Force data saved to: {ft_npy_filename} ({len(self.ft_data_list)} frames)")
            else: rospy.logwarn("No Force data to save.")

            # 2. Save XELA data
            if self.xela_data_list:
                xela_npy_filename = os.path.join(self.current_session_dir, 'xela_data.npy')
                np.save(xela_npy_filename, self.xela_data_list, allow_pickle=True)
                rospy.loginfo(f"✅ XELA data saved to: {xela_npy_filename} ({len(self.xela_data_list)} frames)")
                
                # Also save a JSON version of XELA for inspection
                json_filename = os.path.join(self.current_session_dir, 'xela_data.json')
                with open(json_filename, 'w') as f:
                    json.dump(list(self.xela_data_list), f, indent=2)
            else: rospy.logwarn("No XELA data to save.")

    def start_recording(self):
        """Start a new recording session."""
        if self.recording:
            rospy.logwarn("Already recording!")
            return
        
        self.session_count += 1
        self.current_episode_num = self._get_next_episode_number()
        session_name = f"episode_{self.current_episode_num}"
        self.current_session_dir = os.path.join(self.save_dir, session_name)
        
        os.makedirs(self.current_session_dir, exist_ok=True)
        
        # Create session metadata file
        metadata_file = os.path.join(self.current_session_dir, 'session_info.json')
        metadata = {
            'session_id': self.session_count,
            'episode_num': self.current_episode_num,
            'start_time': datetime.now().isoformat(),
            'ft_topic': self.ft_topic,
            'xela_topic': self.xela_topic,
            'data_format': 'npy',
            'recording_rate_hz': self.rate_hz
        }
        with open(metadata_file, 'w') as f: json.dump(metadata, f, indent=2)
            
        # Reset counters and data lists
        with self.lock:
            self.ft_frame_count = 0
            self.xela_frame_count = 0
            self.ft_data_list = []
            self.xela_data_list = []
            
        self.recording = True
        
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo(f"🔴 RECORDING STARTED | EPISODE {self.current_episode_num}")
        rospy.loginfo(f"Session dir: {self.current_session_dir}")
        rospy.loginfo(f"Collecting at {self.rate_hz} Hz...")
        rospy.loginfo("="*60 + "\n")

    def stop_recording(self):
        """Stop the current recording session and save data."""
        if not self.recording:
            rospy.logwarn("Not currently recording!")
            return
        
        self.recording = False
        
        # Save all accumulated data
        self.save_data_to_npy()
        
        # Update metadata with end time and frame counts
        if self.current_session_dir:
            metadata_file = os.path.join(self.current_session_dir, 'session_info.json')
            try:
                with open(metadata_file, 'r') as f: metadata = json.load(f)
                
                metadata['end_time'] = datetime.now().isoformat()
                metadata['total_force_frames'] = self.ft_frame_count
                metadata['total_xela_frames'] = self.xela_frame_count
                
                with open(metadata_file, 'w') as f: json.dump(metadata, f, indent=2)
            except Exception as e: rospy.logerr(f"Error updating metadata: {e}")
        
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo(f"⏹ RECORDING STOPPED")
        rospy.loginfo(f"Episode: {self.current_episode_num}")
        rospy.loginfo(f"Force frames saved: {self.ft_frame_count}")
        rospy.loginfo(f"XELA frames saved: {self.xela_frame_count}")
        rospy.loginfo("="*60 + "\n")
        
        self.current_session_dir = None
        self.current_episode_num = 0
        
    def pause_recording(self):
        """Toggle pause/resume recording."""
        if not self.current_session_dir:
            rospy.logwarn("No active session to pause/resume! Press 's' to start one.")
            return
        
        self.recording = not self.recording
        
        if self.recording:
            rospy.loginfo("▶️ RECORDING RESUMED")
        else:
            rospy.loginfo("⏸ RECORDING PAUSED")

    # --- Main Loop ---
    
    def run_recording_loop(self):
        """Dedicated loop to collect data at the specified rate."""
        while not rospy.is_shutdown():
            if self.recording:
                # Sample the latest received data and accumulate it
                self._record_and_log_frame()
            
            # This enforces the 10 Hz rate
            self.rate.sleep()

    def run(self):
        """Main run function that starts the recording and keyboard threads."""
        
        # 1. Start the instant keyboard handler thread
        keyboard_handler = KeyboardInputHandler(self)
        try:
            keyboard_handler.start()
            
            # 2. Start the 10Hz data collection thread
            rospy.loginfo("--- Starting Data Collection Thread ---")
            recording_thread = threading.Thread(target=self.run_recording_loop)
            recording_thread.daemon = True
            recording_thread.start()

            # 3. Keep the main ROS thread alive
            rospy.spin()

        except rospy.ROSInterruptException:
            pass
        finally:
            # 4. Cleanup
            if self.recording:
                self.stop_recording()
            keyboard_handler.stop()
            rospy.loginfo("Force-XELA Recorder cleanup complete.")


def main():
    # Check for modules necessary for instant input
    try:
        if not (hasattr(sys.stdin, 'fileno') and sys.stdin.isatty()):
            # This is fine if the user is piping output, but we skip keyboard input setup.
            pass
        # Check if termios and tty are available (Linux/Unix-only check)
        import termios, tty, select
        
    except ImportError:
        print("ERROR: Missing required modules (termios, tty, select). This feature is typically only supported on Linux/Unix systems.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed TTY check: {e}")
        # Let it proceed without instant input if possible, but for this code, it's essential.
        # Keeping the exit for robustness against environment issues.
        sys.exit(1)

    try:
        recorder = ForceXelaRecorder()
        recorder.run() 
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Main Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()