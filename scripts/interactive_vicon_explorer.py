# Updated: 2026-01-29
#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
import time
import glob
import argparse
import json

class ViconVisualizer:
    def __init__(self):
        # Configuration
        self.robot_names = ['ep03', 'ep05']  # Focus on these robots
        self.workstation_names = ['AS_1_neu', 'AS_3_neu', 'AS_4_neu', 'AS_5_neu', 'AS_6_neu']
        self.frame_count = 0
        self.current_frame = 0
        self.playing = False
        self.animation_speed = 1.0
        
        # Data storage
        self.frames = []  # List of frames with timestamp and data
        self.workstation_positions = {}  # Positions by timestamp
        
        # Visualization components
        self.fig = None
        self.ax = None
        self.slider = None
        self.time_text = None
        self.file_text = None
        self.robot_markers = {}
        self.robot_trails = {}
        self.workstation_patches = {}
        self.trail_length = 50
        
        # Robot colors
        self.colors = {
            'ep03': 'blue',
            'ep05': 'red',
        }
        
        # Workstation colors
        self.workstation_colors = {
            'AS_1_neu': 'lightcoral',
            'AS_3_neu': 'lightblue',
            'AS_4_neu': 'lightgreen',
            'AS_5_neu': 'lightyellow',
            'AS_6_neu': 'lightpink'
        }
        
        # Warehouse dimensions
        self.warehouse_bounds = {
            'x_min': -9.1, 'x_max': 10.2,
            'y_min': -4.42, 'y_max': 5.5,
        }
        
        # Default workstation dimensions
        self.workstation_size = {'width': 1.0, 'height': 0.6}
        
    def parse_vicon_file(self, file_path):
        """Parse Vicon data file to extract robot and workstation positions"""
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Data storage by timestamp
        frame_data = {}
        workstation_data = {}
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        # Parse line content
                        data = eval(line.strip())
                        
                        # Skip empty data
                        if all(all(v == 0 for v in obj_data.values()) for obj_data in data.values()):
                            continue
                        
                        # Process objects in this frame
                        timestamp = None
                        robots = {}
                        workstations = {}
                        
                        for key, values in data.items():
                            # Skip empty entries
                            if all(v == 0 for v in values.values()):
                                continue
                                
                            # Extract object name
                            object_name = key.split('/')[1]
                            
                            # Get timestamp (convert to seconds)
                            if timestamp is None:
                                timestamp = values['system_time'] / 1e9
                            
                            # Store position data
                            position_data = {
                                'x': values['pos_x'],
                                'y': values['pos_y'],
                                'z': values['pos_z'],
                                'yaw': values['yaw'],
                                'rot_x': values['rot_x'],
                                'rot_y': values['rot_y'],
                                'rot_z': values['rot_z']
                            }
                            
                            # Sort into robot vs workstation
                            if object_name.startswith('AS_'):
                                workstations[object_name] = position_data
                            elif object_name in self.robot_names:
                                robots[object_name] = position_data
                        
                        # Store frame if it has valid data
                        if timestamp and (robots or workstations):
                            # Add to frames
                            if robots:
                                if timestamp not in frame_data:
                                    frame_data[timestamp] = {}
                                frame_data[timestamp].update(robots)
                            
                            # Add to workstation data
                            if workstations:
                                if timestamp not in workstation_data:
                                    workstation_data[timestamp] = {}
                                workstation_data[timestamp].update(workstations)
                            
                    except Exception as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
        
        # Check if we found any valid data
        if not frame_data:
            print("No valid robot data found!")
            return False
            
        # Sort frames by timestamp
        sorted_times = sorted(frame_data.keys())
        self.frames = [{'timestamp': t, 'robots': frame_data[t]} for t in sorted_times]
        self.frame_count = len(self.frames)
        
        # Pre-process workstation positions
        self.process_workstation_positions(workstation_data)
        
        # Calculate robot trails
        self.build_robot_trails()
        
        print(f"Processed {self.frame_count} frames")
        print(f"Found robots: {', '.join(set().union(*[frame['robots'].keys() for frame in self.frames]))}")
        print(f"Found workstations: {', '.join(self.workstation_positions.keys())}")
        
        return True
    
    def process_workstation_positions(self, workstation_data):
        """Process and organize workstation position data"""
        # First check for dedicated workstation files
        ws_positions_dir = "/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/CPPS_Static_Scenario/archiv_measurements/Vicon/working_station_positions"
        
        if os.path.exists(ws_positions_dir):
            print(f"Searching for workstation positions in {ws_positions_dir}")
            ws_files = glob.glob(os.path.join(ws_positions_dir, "*.txt"))
            
            if ws_files:
                # Sort files by timestamp (as embedded in filenames)
                ws_files.sort(key=lambda x: os.path.getmtime(x))
                
                # Find the file with the closest timestamp to our vicon data
                if self.frames:
                    vicon_start_time = self.frames[0]['timestamp']
                    closest_file = None
                    min_time_diff = float('inf')
                    
                    for ws_file in ws_files:
                        file_time = os.path.getmtime(ws_file)
                        time_diff = abs(file_time - vicon_start_time)
                        
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_file = ws_file
                    
                    if closest_file:
                        print(f"Using workstation positions from {os.path.basename(closest_file)}")
                        workstation_positions = self.parse_workstation_file(closest_file)
                        
                        if workstation_positions:
                            self.workstation_positions = workstation_positions
                            return
                        
            print("No suitable workstation files found, checking vicon data...")
        
        # If no workstation files found or parsing failed, fall back to vicon data
        # If no workstation data
        if not workstation_data:
            print("No workstation data found, using defaults")
            # Default positions as fallback
            self.workstation_positions = {
                'AS_1_neu': {'x': 1.52, 'y': 2.24, 'z': 1.02, 'yaw': 0},
                'AS_3_neu': {'x': -5.74, 'y': -0.13, 'z': 1.47, 'yaw': 0},
                'AS_4_neu': {'x': 5.37, 'y': 0.21, 'z': 2.30, 'yaw': np.pi/2},  # 90 degree rotation
                'AS_5_neu': {'x': -3.05, 'y': 2.39, 'z': 2.21, 'yaw': 0},
                'AS_6_neu': {'x': 0.01, 'y': -1.45, 'z': 1.53, 'yaw': 0}
            }
            return
        
        # Calculate average positions for each workstation
        avg_positions = {}
        
        for ws_name in self.workstation_names:
            positions = []
            
            # Collect all positions for this workstation
            for timestamp, ws_data in workstation_data.items():
                if ws_name in ws_data:
                    positions.append(ws_data[ws_name])
            
            if positions:
                # Calculate average position
                avg_pos = {
                    'x': np.mean([p['x'] for p in positions]),
                    'y': np.mean([p['y'] for p in positions]),
                    'z': np.mean([p['z'] for p in positions]),
                    'yaw': np.mean([p['yaw'] for p in positions])
                }
                
                # Apply 90-degree rotation to AS_4_neu
                if ws_name == 'AS_4_neu':
                    avg_pos['yaw'] = np.pi/2  # 90 degrees in radians
                
                avg_positions[ws_name] = avg_pos
                print(f"Average position for {ws_name}: ({avg_pos['x']:.2f}, {avg_pos['y']:.2f})")
        
        self.workstation_positions = avg_positions

    def parse_workstation_file(self, file_path):
        """Parse a workstation positions file"""
        try:
            workstation_positions = {}
            
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        # Parse line content
                        data = eval(line.strip())
                        
                        # Process each object in the data
                        for key, values in data.items():
                            # Extract object name 
                            object_name = key.split('/')[1]
                            
                            # Only process workstations
                            if object_name.startswith('AS_'):
                                # Store position data
                                workstation_positions[object_name] = {
                                    'x': values['pos_x'],
                                    'y': values['pos_y'],
                                    'z': values['pos_z'],
                                    'yaw': values['yaw'],
                                    'rot_x': values['rot_x'],
                                    'rot_y': values['rot_y'],
                                    'rot_z': values['rot_z']
                                }
                                
                                # Apply 90-degree rotation to AS_4_neu
                                if object_name == 'AS_4_neu':
                                    workstation_positions[object_name]['yaw'] = np.pi/2
                                
                                print(f"Found workstation {object_name} at ({values['pos_x']:.2f}, {values['pos_y']:.2f})")
                        
                    except Exception as e:
                        print(f"Error parsing line in workstation file: {e}")
                        continue
            
            return workstation_positions
        
        except Exception as e:
            print(f"Error reading workstation file: {e}")
            return None
            
   
    def build_robot_trails(self):
        """Pre-calculate position trails for robots"""
        print("Building robot trails...")
        
        # For each robot, collect positions over time
        robot_positions = {robot: [] for robot in self.robot_names}
        
        # Extract position data
        for frame in self.frames:
            for robot, data in frame['robots'].items():
                robot_positions[robot].append((data['x'], data['y']))
        
        # Pre-compute trails for each frame
        for i in range(self.frame_count):
            frame = self.frames[i]
            frame['trails'] = {}
            
            for robot in self.robot_names:
                positions = robot_positions[robot][:i+1]
                
                if positions:
                    # Use last N positions for trail
                    trail = positions[-self.trail_length:] if len(positions) > self.trail_length else positions
                    frame['trails'][robot] = trail
    
    def setup_visualization(self, title="Vicon Data Visualization"):
        """Set up the visualization figure and controls"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.subplots_adjust(bottom=0.25)
        
        # Set up axes
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Draw warehouse boundaries
        self.draw_warehouse()
        
        # Initialize workstation patches
        self.initialize_workstations()
        
        # Initialize robot markers
        self.initialize_robots()
        
        # Time display
        self.time_text = self.ax.text(0.02, 0.95, "Time: 0.00s", 
                                     transform=self.ax.transAxes, fontsize=10,
                                     bbox=dict(facecolor='wheat', alpha=0.7))
        
        # Add slider for time control
        slider_ax = self.fig.add_axes([0.2, 0.1, 0.65, 0.03])
        self.slider = Slider(
            slider_ax, 'Time', 0, max(1, self.frame_count-1),
            valinit=0, valstep=1, color='green'
        )
        self.slider.on_changed(self.on_slider_change)
        
        # Play/pause button
        play_ax = self.fig.add_axes([0.05, 0.1, 0.1, 0.03])
        self.play_button = Button(play_ax, 'Play', color='lightgoldenrodyellow')
        self.play_button.on_clicked(self.toggle_play)
        
        # Speed control buttons
        speed_slower_ax = self.fig.add_axes([0.05, 0.05, 0.1, 0.03])
        speed_faster_ax = self.fig.add_axes([0.2, 0.05, 0.1, 0.03])
        self.speed_slower = Button(speed_slower_ax, 'Slower', color='lightblue')
        self.speed_faster = Button(speed_faster_ax, 'Faster', color='lightblue')
        self.speed_slower.on_clicked(self.decrease_speed)
        self.speed_faster.on_clicked(self.increase_speed)
        
        # Add legend
        self.ax.legend(loc='upper right')
    
    def draw_warehouse(self):
        """Draw warehouse boundaries"""
        # Plot boundary
        warehouse_x = [
            self.warehouse_bounds['x_min'], self.warehouse_bounds['x_max'],
            self.warehouse_bounds['x_max'], self.warehouse_bounds['x_min'],
            self.warehouse_bounds['x_min']
        ]
        warehouse_y = [
            self.warehouse_bounds['y_min'], self.warehouse_bounds['y_min'],
            self.warehouse_bounds['y_max'], self.warehouse_bounds['y_max'],
            self.warehouse_bounds['y_min']
        ]
        self.ax.plot(warehouse_x, warehouse_y, 'k-', linewidth=2)
        
        # Set axis limits with margin
        margin = 1.0
        self.ax.set_xlim(
            self.warehouse_bounds['x_min'] - margin,
            self.warehouse_bounds['x_max'] + margin
        )
        self.ax.set_ylim(
            self.warehouse_bounds['y_min'] - margin,
            self.warehouse_bounds['y_max'] + margin
        )
    
    def initialize_workstations(self):
        """Initialize workstation visualization elements"""
        for ws_name, pos in self.workstation_positions.items():
            color = self.workstation_colors.get(ws_name, 'lightgray')
            width = self.workstation_size['width']
            height = self.workstation_size['height']
            
            # Create rectangle at initial position
            # Note: Will be updated later based on orientation
            rect = Rectangle(
                (pos['x'] - width/2, pos['y'] - height/2),
                width, height,
                facecolor=color,
                edgecolor='black',
                alpha=0.7,
                angle=0  # Will be rotated based on yaw
            )
            self.ax.add_patch(rect)
            
            # Store patch reference
            self.workstation_patches[ws_name] = {
                'rect': rect,
                'label': self.ax.text(
                    pos['x'], pos['y'], ws_name,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8
                )
            }
            
            # Apply initial rotation for AS_4
            if ws_name == 'AS_4_neu':
                # For matplotlib Rectangle, we need to update the xy position 
                # when changing angle to keep the center fixed
                center_x, center_y = pos['x'], pos['y']
                corner_x = center_x - width/2
                corner_y = center_y - height/2
                
                # For 90-degree rotation, swap width and height
                rect.set_width(height)
                rect.set_height(width)
                rect.set_xy((center_x - height/2, center_y - width/2))
    
    def initialize_robots(self):
        """Initialize robot visualization elements"""
        for robot in self.robot_names:
            color = self.colors.get(robot, 'gray')
            
            # Position marker
            marker = self.ax.scatter([], [], s=80, color=color, 
                                    edgecolor='black', label=robot)
            
            # Direction arrow
            arrow = self.ax.quiver([], [], [], [], color=color, 
                                  scale=5, width=0.008, zorder=10)
            
            # Position text
            text = self.ax.text(0, 0, '', fontsize=9, 
                              bbox=dict(facecolor='white', alpha=0.7))
            
            # Trail line
            trail, = self.ax.plot([], [], '-', color=color, alpha=0.6, linewidth=1.5)
            
            # Store visualization objects
            self.robot_markers[robot] = {
                'marker': marker,
                'arrow': arrow,
                'text': text,
                'trail': trail
            }
    
    def update_visualization(self, frame_idx=None):
        """Update visualization to show the specified frame"""
        if not self.frames:
            return
        
        # Use provided index or current frame
        if frame_idx is not None:
            self.current_frame = frame_idx
        
        # Get frame data
        frame = self.frames[self.current_frame]
        robots = frame['robots']
        timestamp = frame['timestamp']
        trails = frame['trails']
        
        # Update time display (relative to first frame)
        rel_time = timestamp - self.frames[0]['timestamp']
        self.time_text.set_text(f"Time: {rel_time:.2f}s")
        
        # Update robot positions
        for robot, markers in self.robot_markers.items():
            if robot in robots:
                # Get position data
                pos = robots[robot]
                x, y = pos['x'], pos['y']
                yaw = pos['yaw']
                
                # Update position marker
                markers['marker'].set_offsets([[x, y]])
                
                # Update direction arrow
                arrow_len = 0.4
                dx = arrow_len * np.cos(yaw)
                dy = arrow_len * np.sin(yaw)
                markers['arrow'].set_offsets([[x, y]])
                markers['arrow'].set_UVC([dx], [dy])
                
                # Update position text
                markers['text'].set_position((x + 0.3, y + 0.3))
                markers['text'].set_text(
                    f"{robot}\n"
                    f"X: {x:.2f}\n"
                    f"Y: {y:.2f}"
                )
                
                # Update trail
                if robot in trails and trails[robot]:
                    trail_x, trail_y = zip(*trails[robot])
                    markers['trail'].set_data(trail_x, trail_y)
                else:
                    markers['trail'].set_data([], [])
            else:
                # Robot not in this frame, hide elements
                markers['marker'].set_offsets([[]])
                markers['arrow'].set_UVC([], [])
                markers['text'].set_text('')
                markers['trail'].set_data([], [])
        
        # Redraw canvas
        self.fig.canvas.draw_idle()
    
    def on_slider_change(self, val):
        """Handle slider value change"""
        frame_idx = int(val)
        self.update_visualization(frame_idx)
    
    def toggle_play(self, event=None):
        """Toggle animation playback"""
        self.playing = not self.playing
        
        if self.playing:
            self.play_button.label.set_text('Pause')
            self.animate()
        else:
            self.play_button.label.set_text('Play')
    
    def animate(self):
        """Animation loop with real-time playback"""
        if not self.playing:
            return
        
        # If we're at the end, loop back to start
        if self.current_frame >= self.frame_count - 1:
            self.current_frame = 0
            self.slider.set_val(self.current_frame)
            return
        
        # Calculate current and next timestamps
        current_time = self.frames[self.current_frame]['timestamp']
        next_frame = self.current_frame + 1
        next_time = self.frames[next_frame]['timestamp']
        
        # Calculate real time difference between frames
        time_diff = (next_time - current_time) / self.animation_speed
        
        # Advance frame
        self.current_frame = next_frame
        
        # Update slider (this will trigger visualization update)
        self.slider.set_val(self.current_frame)
        
        # Schedule next frame update with the real-time delay
        if self.playing:
            # Ensure minimum frame time to avoid excessive CPU usage
            time_diff = max(0.01, time_diff)  # Minimum 10ms delay
            self.fig.canvas.start_event_loop(time_diff)
            self.fig.canvas.mpl_connect('draw_event', self._on_draw)
    
    def _on_draw(self, event):
        """Handle draw event to continue animation"""
        if self.playing:
            self.animate()
    
    def increase_speed(self, event):
        """Increase animation speed"""
        self.animation_speed = min(4.0, self.animation_speed * 1.5)
        print(f"Animation speed: {self.animation_speed:.1f}x")
    
    def decrease_speed(self, event):
        """Decrease animation speed"""
        self.animation_speed = max(0.25, self.animation_speed / 1.5)
        print(f"Animation speed: {self.animation_speed:.1f}x")
    
    def run_visualization(self, file_path):
        """Run the full visualization process for a file"""
        # Parse Vicon data
        if not self.parse_vicon_file(file_path):
            print("Failed to parse Vicon data!")
            return False
        
        # Setup visualization
        title = f"Robot Positions - {os.path.basename(file_path)}"
        self.setup_visualization(title)
        
        # Initial frame update
        self.update_visualization(0)
        
        # Start visualization
        plt.show()
        return True

def main():
    parser = argparse.ArgumentParser(description='Interactive visualization of Vicon data with slider')
    parser.add_argument('file', help='Vicon data file to visualize')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    # Create visualizer and run
    viz = ViconVisualizer()
    viz.run_visualization(args.file)

if __name__ == "__main__":
    main()