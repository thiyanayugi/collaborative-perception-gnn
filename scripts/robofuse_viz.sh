#!/bin/bash

# Script to automate ROS bag visualization for RoboFUSE Dataset
# Save this as ~/robofuse_viz.sh and make it executable with: chmod +x ~/robofuse_viz.sh

# Base directory for dataset
BASE_DIR="/media/yugi/MAIN DRIVE/RoboFUSE_Dataset/CPPS_Static_Scenario"

# Use existing RViz configurations
RVIZ_CONFIG_ROBOT1="$HOME/curr.rviz"
RVIZ_CONFIG_ROBOT2="$HOME/curr2.rviz"


# Function to show available scenarios
show_scenarios() {
    echo "Available scenarios:"
    echo "1) CPPS_Horizontal"
    echo "2) CPPS_Vertical"
    echo "3) CPPS_Diagonal"
    echo "4) CPPS_Horizontal_Vertical"
    echo "5) CPPS_Horizontal_Diagonal"
    echo "6) CPPS_Vertical_Horizontal"
    echo "7) CPPS_Diagonal_Horizontal"
    echo "0) Exit"
}

# Function to show available robots and recordings for a scenario
show_robots() {
    scenario=$1
    
    # Check if Robot_1 directory exists
    if [ -d "$BASE_DIR/$scenario/Robot_1/rosbag" ]; then
        echo "Robot_1 (ep03) recordings:"
        count=1
        for bag in "$BASE_DIR/$scenario/Robot_1/rosbag"/*; do
            if [ -d "$bag" ] && [ -f "$bag"/*.db3 ]; then
                echo "  $count) $(basename "$bag")"
                count=$((count+1))
            fi
        done
    fi
    
    echo ""
    
    # Check if Robot_2 directory exists
    if [ -d "$BASE_DIR/$scenario/Robot_2/rosbag" ]; then
        echo "Robot_2 (ep05) recordings:"
        count=1
        for bag in "$BASE_DIR/$scenario/Robot_2/rosbag"/*; do
            if [ -d "$bag" ] && [ -f "$bag"/*.db3 ]; then
                echo "  $count) $(basename "$bag")"
                count=$((count+1))
            fi
        done
    fi
}

# Function to start TF publishers
start_tf_publishers() {
    robot_id=$1
    
    # Kill any existing TF publishers
    pkill -f "static_transform_publisher" || true
    
    # Start new TF publishers based on robot ID
    if [ "$robot_id" == "ep03" ]; then
        ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map ep03/odom &
        ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 ep03/odom base_link &
        ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link ep03/wave_sensor_link &
    elif [ "$robot_id" == "ep05" ]; then
        ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map ep05/odom &
        ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 ep05/odom base_link &
        ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link ep05/wave_sensor_link &
    fi
    
    echo "TF publishers started for $robot_id"
}

# Function to play a ROS bag
play_bag() {
    scenario=$1
    robot=$2
    recording=$3
    playback_rate=$4
    
    # Determine robot ID
    robot_id=$([ "$robot" == "Robot_1" ] && echo "ep03" || echo "ep05")
    
    # Get the nth recording
    count=1
    selected_bag=""
    for bag in "$BASE_DIR/$scenario/$robot/rosbag"/*; do
        if [ -d "$bag" ] && [ -f "$bag"/*.db3 ]; then
            if [ $count -eq $recording ]; then
                selected_bag="$bag"
                break
            fi
            count=$((count+1))
        fi
    done
    
    if [ -z "$selected_bag" ]; then
        echo "Recording not found!"
        return
    fi
    
    echo "Playing bag: $selected_bag"
    echo "Starting TF publishers for $robot_id..."
    ros2 param set /use_sim_time true
    ros2 param set /rviz2 use_sim_time true


    
    # Start TF publishers in the current terminal
    start_tf_publishers "$robot_id"
    
    echo "Would you like to: "
    echo "1) Start RViz2 now"
    echo "2) Skip RViz2 (if it's causing problems)"
    read -p "Enter choice (1-2): " rviz_choice
    
    if [ "$rviz_choice" -eq 1 ]; then
        echo "Starting RViz2 in a new terminal window..."
        
	       # Start RViz in a new terminal tab
	if [ "$robot_id" == "ep03" ] && [ -f "$RVIZ_CONFIG_ROBOT1" ]; then
	    gnome-terminal --tab -- bash -c "source /opt/ros/humble/setup.bash; \
		            export LIBGL_ALWAYS_SOFTWARE=1; \
		            export ROS_TF_BUFFER_CACHE_TIME_NS=10000000000; \
		            export ROS_TF_FILTER_QUEUE_SIZE=1000; \
		            export ROS_NAMESPACE=/ep03; \
		            rviz2 -d $RVIZ_CONFIG_ROBOT1; \
		            exec bash" &
	elif [ "$robot_id" == "ep05" ] && [ -f "$RVIZ_CONFIG_ROBOT2" ]; then
	    gnome-terminal --tab -- bash -c "source /opt/ros/humble/setup.bash; \
		            export LIBGL_ALWAYS_SOFTWARE=1; \
		            export ROS_TF_BUFFER_CACHE_TIME_NS=10000000000; \
		            export ROS_TF_FILTER_QUEUE_SIZE=1000; \
		            export ROS_NAMESPACE=/ep05; \
		            rviz2 -d $RVIZ_CONFIG_ROBOT2; \
		            exec bash" &
	else
	    echo "Warning: RViz configuration not found. Starting with default config."
	    gnome-terminal --tab -- bash -c "source /opt/ros/humble/setup.bash; \
		            export LIBGL_ALWAYS_SOFTWARE=1; \
		            export ROS_TF_BUFFER_CACHE_TIME_NS=10000000000; \
		            export ROS_TF_FILTER_QUEUE_SIZE=1000; \
		            export ROS_NAMESPACE=/$robot_id; \
		            rviz2; \
		            exec bash" &
	fi
        # Give time for RViz to start up
        sleep 2
    fi
    
    # Play the ROS bag
    echo "Starting bag playback. Press Ctrl+C to stop."
    ros2 bag play --loop "$selected_bag" --clock -r "$playback_rate" --read-ahead-queue-size 1000

}

# Check if ROS 2 is sourced
if ! command -v ros2 &> /dev/null; then
    echo "ROS 2 commands not found. Sourcing ROS 2..."
    source /opt/ros/humble/setup.bash
fi

# Main menu
while true; do
    echo ""
    echo "RoboFUSE Visualization Tool"
    echo "--------------------------"
    
    show_scenarios
    echo ""
    read -p "Select a scenario (0-7): " scenario_choice
    
    case $scenario_choice in
        0)
            echo "Exiting..."
            # Clean up before exiting
            pkill -f "static_transform_publisher" || true
            exit 0
            ;;
        1)
            scenario="CPPS_Horizontal"
            ;;
        2)
            scenario="CPPS_Vertical"
            ;;
        3)
            scenario="CPPS_Diagonal"
            ;;
        4)
            scenario="CPPS_Horizontal_Vertical"
            ;;
        5)
            scenario="CPPS_Horizontal_Diagonal"
            ;;
        6)
            scenario="CPPS_Vertical_Horizontal"
            ;;
        7)
            scenario="CPPS_Diagonal_Horizontal"
            ;;
        *)
            echo "Invalid choice. Please try again."
            continue
            ;;
    esac
    
    echo ""
    echo "Selected scenario: $scenario"
    echo ""
    
    show_robots "$scenario"
    echo ""
    
    read -p "Select robot (1 or 2): " robot_choice
    if [ "$robot_choice" -eq 1 ]; then
        robot="Robot_1"
    elif [ "$robot_choice" -eq 2 ]; then
        robot="Robot_2"
    else
        echo "Invalid choice. Please try again."
        continue
    fi
    
    echo ""
    read -p "Select recording number: " recording_choice
    
    echo ""
    read -p "Enter playback rate (e.g., 0.5 for half speed, 1.0 for normal): " playback_rate
    playback_rate=${playback_rate:-0.5}  # Default to 0.5 if not specified
    
    # Play the selected ROS bag
    play_bag "$scenario" "$robot" "$recording_choice" "$playback_rate"
    
    # Clean up TF publishers when returning to menu
    pkill -f "static_transform_publisher" || true
done
