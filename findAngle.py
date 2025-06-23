import numpy as np
from sympy.solvers import solve
from sympy import Symbol
import os
import pandas as pd
import re
from math import acos, degrees
from pyk4a import PyK4APlayback

csv_file_path = './timings.csv'  
movements = pd.read_csv(csv_file_path)

frame_rate = 30


joints = ["PELVIS",	
	"SPINE_NAVAL",	
	"SPINE_CHEST",
	"NECK",
	"CLAVICLE_LEFT",
	"SHOULDER_LEFT", 
	"ELBOW_LEFT",
	"WRIST_LEFT",
	"HAND_LEFT",
	"HANDTIP_LEFT",
	"THUMB_LEFT",
	"CLAVICLE_RIGHT",
	"SHOULDER_RIGHT",
	"ELBOW_RIGHT",	
	"WRIST_RIGHT",	
	"HAND_RIGHT",
	"HANDTIP_RIGHT",
	"THUMB_RIGHT",	
	"HIP_LEFT",
	"KNEE_LEFT",	
	"ANKLE_LEFT",	
	"FOOT_LEFT",
	"HIP_RIGHT",
	"KNEE_RIGHT",
	"ANKLE_RIGHT",
	"FOOT_RIGHT",
	"HEAD",
	"NOSE",	
	"EYE_LEFT",
	"EAR_LEFT",
	"EYE_RIGHT",
	"EAR_RIGHT"]

def get_coordinates(video_path):
    jointNum = 32
    frames = []
    frame = []
    with open(video_path, "r") as coordinates:
        content = coordinates.read().strip()
        coordinate_lists = content.strip("][").split("][")

        for coordinate_list in coordinate_lists:
            frame.append(coordinate_list)
            if len(frame) == jointNum:
                frames.append(frame)
                frame = []
    return frames


def get_movement_start_end_frames(stripped_video_name):
    movement_start_frame = None
    movement_end_frame = None
    movement_start_frame2 = None
    movement_end_frame2 = None
    start_second = None
    end_second = None
    video_row_num = None

    row_mask = movements['Videos'] == stripped_video_name
    if not row_mask.any():
        print(f"{stripped_video_name} not found.")
        return None, None, None, None, None
    
    pattern = r"^.*_(.*?)_[A-Z]{2}_.*"
    joint_name = re.sub(pattern, r"\1", stripped_video_name)
    video_row_num = row_mask.idxmax()  # first index where mask is True
    print("File found in:", video_row_num + 2, stripped_video_name)
    row_data = movements.loc[video_row_num]

    if "BACK" in stripped_video_name:
        if pd.notna(row_data["BACK__BEND_FORWARD"]) and pd.notna(row_data["BACK__BEND_BACKWARD"]):
            start_second = list(map(int, row_data["BACK__BEND_FORWARD"].strip("()").split(",")))[0]
            movement_start_frame = start_second * frame_rate
            end_second = list(map(int, row_data["BACK__BEND_BACKWARD"].strip("()").split(",")))[1]
            movement_end_frame = end_second * frame_rate
        elif pd.notna(row_data["BACK__BEND_FORWARD_AND_BACKWARD"]):
            start_second, end_second = map(int, row_data["BACK__BEND_FORWARD_AND_BACKWARD"].strip("()").split(","))
            movement_start_frame = start_second * frame_rate
            movement_end_frame = end_second * frame_rate
        if pd.notna(row_data["BACK__BEND_LEFT_AND_RIGHT"]):
            start_second, end_second = map(int, row_data["BACK__BEND_LEFT_AND_RIGHT"].strip("()").split(","))
            movement_start_frame2 = start_second * frame_rate
            movement_end_frame2 = end_second * frame_rate
    elif "NECK" in stripped_video_name:
        if pd.notna(row_data["NECK__LOOK_UP_AND_DOWN"]):
            start_second, end_second = map(int, row_data["NECK__LOOK_UP_AND_DOWN"].strip("()").split(","))
            movement_start_frame = start_second * frame_rate
            movement_end_frame = end_second * frame_rate
    elif "SHOULDERS" in stripped_video_name:
        if pd.notna(row_data["SHOULDER__RAISE_ARMS_OUTWARD__BOTH"]):
            start_second, end_second = map(int, row_data["SHOULDER__RAISE_ARMS_OUTWARD__BOTH"].strip("()").split(","))
            movement_start_frame = start_second * frame_rate
            movement_end_frame = end_second * frame_rate
        elif pd.notna(row_data["SHOULDER__RAISE_ARMS_IN_FRONT__BOTH"]):
            start_second, end_second = map(int, row_data["SHOULDER__RAISE_ARMS_IN_FRONT__BOTH"].strip("()").split(","))
            movement_start_frame = start_second * frame_rate
            movement_end_frame = end_second * frame_rate
    elif "WRISTS" in stripped_video_name:
        if pd.notna(row_data["WRIST__BEND_LEFT_AND_RIGHT__BOTH"]):
            start_second, end_second = map(int, row_data["WRIST__BEND_LEFT_AND_RIGHT__BOTH"].strip("()").split(","))
            movement_start_frame = start_second * frame_rate
            movement_end_frame = end_second * frame_rate
    elif "HIP" in stripped_video_name:
        if pd.notna(row_data["BACK__BEND_LEFT_AND_RIGHT"]):
            start_second, end_second = map(int, row_data["BACK__BEND_LEFT_AND_RIGHT"].strip("()").split(","))
            movement_start_frame = start_second * frame_rate
            movement_end_frame = end_second * frame_rate
    # else: 
    #     for col in row_data.index[1:]:
    #         if joint_name in col and pd.notna(row_data[col]):
    #             print(f"Movement found: {col}")
    #             start_second, end_second = map(int, row_data[col].strip("()").split(","))
    #             movement_start_frame = start_second * frame_rate
    #             movement_end_frame = end_second * frame_rate
    #             break
    
    if movement_start_frame is None or movement_end_frame is None:
        for col in row_data.index[1:]:
            if joint_name in col and pd.notna(row_data[col]):
                print(f"Movement found: {col}")
                start_second, end_second = map(int, row_data[col].strip("()").split(","))
                movement_start_frame = start_second * frame_rate
                movement_end_frame = end_second * frame_rate
                break

    if movement_start_frame is None or movement_end_frame is None:
        print(f"Movement start and end frames not found in csv file for {stripped_video_name}")
    else:
        label = None
        print(movement_start_frame, movement_end_frame)
        pain_frames, fake_frames, painless_frames, rest_frames = get_pain_and_fake_frames()
        print(pain_frames, fake_frames)

        if pain_frames != {}:
            for start_frame, end_frame in pain_frames.items():
                if start_frame <= movement_end_frame and end_frame >= movement_start_frame:
                    label = 0
                    print("Pain")
                    break
        if fake_frames != {}:
            for start_frame, end_frame in fake_frames.items():
                if start_frame <= movement_end_frame and end_frame >= movement_start_frame:
                    label = 1
                    print("Not real")
                    break

        return movement_start_frame, movement_end_frame, start_second, end_second, video_row_num, label, movement_start_frame2, movement_end_frame2
    
   
# Returns start_frame: end_frame dictionary for all classes in New_txt files
def get_pain_and_fake_frames():
    pain_frames = {}
    fake_frames = {}
    painless_frames = {}
    rest_frames = {}
    try:
        with open(rf"D:\New_txt\{stripped_video_name[:-1]}.txt", "r") as f:
            for line in f:
                tokens = line.strip().split(" ")
                if tokens[0] == "Pain":
                    start_frame = int(tokens[1]) - (int(tokens[1]) % frame_rate)
                    end_frame = int(tokens[2]) - (int(tokens[2]) % frame_rate)
                    pain_frames[start_frame] = end_frame
                elif tokens[0] == "Fake":
                    start_frame = int(tokens[1]) - (int(tokens[1]) % frame_rate)
                    end_frame = int(tokens[2]) - (int(tokens[2]) % frame_rate)
                    fake_frames[start_frame] = end_frame
                elif tokens[0] == "Painless":
                    start_frame = int(tokens[1]) - (int(tokens[1]) % frame_rate)
                    end_frame = int(tokens[2]) - (int(tokens[2]) % frame_rate)
                    painless_frames[start_frame] = end_frame
                elif tokens[0] == "Rest":
                    start_frame = int(tokens[1]) - (int(tokens[1]) % frame_rate)
                    end_frame = int(tokens[2]) - (int(tokens[2]) % frame_rate)
                    rest_frames[start_frame] = end_frame
    except Exception as e:
        print(f"Error get_label: {e} in {stripped_video_name[:-1]}")

    return pain_frames, fake_frames, painless_frames, rest_frames

def get_gravity_vector(imu_sample, mkv_path):

    print(f"Get gravity vector for video {mkv_path}")
    if imu_sample is None:
        print("IMU data is not available in this recording.")
        return None
    
    acc_data = imu_sample["acc_sample"]

    # accelerometer data is the reaction force to gravity so gravity is just the negative of the accelerometer data
    gravity_vector = [-acc_data[0], -acc_data[1], -acc_data[2]]

    # accelerometer coordinate system is x+ backward, y+ up, z+ right
    # depth camera coordinate system is x+ right, y+ down, z+ forward
    gravity_vector = [-gravity_vector[1], gravity_vector[2], -gravity_vector[0]]  

    return gravity_vector


    


def calculate_angular_velocity(angles, frame_rate=30):
    angular_velocities = {}
    delta_t = 1 / frame_rate  # Time difference between frames

    previous_angle = None
    for frame_index, angle in angles.items():
        if previous_angle is not None:
            delta_theta = angle - previous_angle
            angular_velocity = delta_theta / delta_t
            angular_velocities[frame_index] = angular_velocity
        
        previous_angle = angle

    return angular_velocities

def calculate_angular_acceleration(angular_velocities, frame_duration= 1 / frame_rate):
    angular_accelerations = np.zeros(len(angular_velocities))
    
    for i in range(1, len(angular_velocities)):
        delta_angular_velocity = angular_velocities[i] - angular_velocities[i - 1]
        angular_accelerations[i] = delta_angular_velocity / frame_duration

    return angular_accelerations

# Direction vector from SPINE_CHEST to PELVIS
def get_dv_chest_pelvis(frame):

    # Read https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints#joint-hierarchy

    chest = np.array(list(map(float, frame[2].strip().split(","))))
    pelvis = np.array(list(map(float, frame[0].strip().split(","))))
    
    dv_chest_pelvis = np.subtract(pelvis, chest)

    return dv_chest_pelvis

def get_wrist_angle(frames, movement_start_frame, movement_end_frame):
    
    angles = {}
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            print(f"Broke in frame {frame_index} of {len(frames)}")
            break
        frame = frames[frame_index]

        left_elbow = np.array(list(map(float, frame[6].strip().split(","))))
        left_wrist = np.array(list(map(float, frame[7].strip().split(","))))

        dv_elbow_wrist = np.subtract(left_wrist, left_elbow)
        dv_chest_pelvis = get_dv_chest_pelvis(frame)

        dot_product = np.dot(dv_chest_pelvis, dv_elbow_wrist)
        norm_chest_pelvis = np.linalg.norm(dv_chest_pelvis)
        norm_elbow_wrist = np.linalg.norm(dv_elbow_wrist)

        angle = np.degrees(np.arccos(dot_product / (norm_chest_pelvis * norm_elbow_wrist)))

        angles[frame_index] = angle

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(list(angular_velocities.values()))
    
    smallest_angle_index, smallest_angle = min(angles.items(), key=lambda item: item[1])
    
    biggest_angle_index, biggest_angle = max(angles.items(), key=lambda item: item[1])
    print(f"Smallest angle: {smallest_angle:.2f} in frame {smallest_angle_index} second {(smallest_angle_index / frame_rate):.2f} ")
    print(f"Biggest angle: {biggest_angle:.2f} in frame {biggest_angle_index} second {(biggest_angle_index / frame_rate):.2f} ")
    print("---------------------------------------")
    return angles, angular_velocities, angular_accelerations

def get_knee_angle(frames, movement_start_frame, movement_end_frame, isLeft):
    
    angles = {}
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            print(f"Broke in frame {frame_index} of {len(frames)}")
            break
        frame = frames[frame_index]
        
        hip = frame[18] if isLeft else frame[22]
        knee = frame[19] if isLeft else frame[23]
        ankle = frame[20] if isLeft else frame[24]
        hip = np.array(list(map(float, hip.strip().split(","))))
        knee = np.array(list(map(float, knee.strip().split(","))))
        ankle = np.array(list(map(float, ankle.strip().split(","))))

        dv_hip_knee = np.subtract(knee, hip)
        dv_knee_ankle = np.subtract(knee, ankle)

        dot_product = np.dot(dv_hip_knee, dv_knee_ankle)
        norm_hip_knee = np.linalg.norm(dv_hip_knee)
        norm_knee_ankle = np.linalg.norm(dv_knee_ankle)

        angle = np.degrees(np.arccos(dot_product / (norm_hip_knee * norm_knee_ankle)))

        angles[frame_index] = angle

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(list(angular_velocities.values()))

    smallest_angle_index, smallest_angle = min(angles.items(), key=lambda item: item[1])
    
    biggest_angle_index, biggest_angle = max(angles.items(), key=lambda item: item[1])
    print(f"Smallest angle: {smallest_angle:.2f} in frame {smallest_angle_index} second {(smallest_angle_index / frame_rate):.2f} ")
    print(f"Biggest angle: {biggest_angle:.2f} in frame {biggest_angle_index} second {(biggest_angle_index / frame_rate):.2f} ")
    print("---------------------------------------")
    return angles, angular_velocities, angular_accelerations


def get_elbow_angle(frames, movement_start_frame, movement_end_frame, isLeft):
    
    angles = {}
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            print(f"Broke in frame {frame_index} of {len(frames)}")
            break
        frame = frames[frame_index]

        shoulder = frame[5] if isLeft else frame[12]
        elbow = frame[6] if isLeft else frame[13]
        wrist = frame[7] if isLeft else frame[14]
        shoulder = np.array(list(map(float, shoulder.strip().split(","))))
        elbow = np.array(list(map(float, elbow.strip().split(","))))
        wrist = np.array(list(map(float, wrist.strip().split(","))))

        dv_shoulder_elbow = np.subtract(elbow, shoulder)
        dv_elbow_wrist = np.subtract(elbow, wrist)

        dot_product = np.dot(dv_shoulder_elbow, dv_elbow_wrist)
        norm_shoulder_elbow = np.linalg.norm(dv_shoulder_elbow)
        norm_elbow_wrist = np.linalg.norm(dv_elbow_wrist)

        angle = np.degrees(np.arccos(dot_product / (norm_shoulder_elbow * norm_elbow_wrist)))

        angles[frame_index] = angle
        
    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(list(angular_velocities.values()))

    smallest_angle_index, smallest_angle = min(angles.items(), key=lambda item: item[1])
    
    biggest_angle_index, biggest_angle = max(angles.items(), key=lambda item: item[1])
    print(f"Smallest angle: {smallest_angle:.2f} in frame {smallest_angle_index} second {(smallest_angle_index / frame_rate):.2f}")
    print(f"Biggest angle: {biggest_angle:.2f} in frame {biggest_angle_index} second {(biggest_angle_index / frame_rate):.2f}")
    print("---------------------------------------")
    return angles, angular_velocities, angular_accelerations

def get_neck_angle(frames, movement_start_frame, movement_end_frame):
    
    angles = {}
    # reflex_angles = {}
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            print(f"Broke in frame {frame_index} of {len(frames)}")
            break
        frame = frames[frame_index]

        head = np.array(list(map(float, frame[26].strip().split(","))))
        nose = np.array(list(map(float, frame[27].strip().split(","))))
        spine_chest = np.array(list(map(float, frame[2].strip().split(","))))
        pelvis = np.array(list(map(float, frame[3].strip().split(","))))

        dv_head_nose = np.subtract(head, nose)
        dv_pelvis_chest = np.subtract(pelvis, spine_chest)

        dot_product = np.dot(dv_pelvis_chest, dv_head_nose)

        angle = np.degrees(np.arccos(dot_product / (np.linalg.norm(dv_pelvis_chest) * np.linalg.norm(dv_head_nose))))

        angles[frame_index] = angle

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(list(angular_velocities.values()))
    smallest_angle_index, smallest_angle = min(angles.items(), key=lambda item: item[1])
    biggest_angle_index, biggest_angle = max(angles.items(), key=lambda item: item[1])
    
    print(f"Smallest angle: {smallest_angle:.2f} in frame {smallest_angle_index} second {(smallest_angle_index / frame_rate):.2f}")
    print(f"Biggest angle: {biggest_angle:.2f} in frame {biggest_angle_index} second {(biggest_angle_index / frame_rate):.2f}")
    print("---------------------------------------")
    return angles, angular_velocities, angular_accelerations


def get_shoulder_angle(frames, movement_start_frame, movement_end_frame, isLeft):
    
    angles = {}

    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            print(f"Broke in frame {frame_index} of {len(frames)}")
            break
        frame = frames[frame_index]
        
        # Read https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints#joint-hierarchy

        shoulder = frame[5] if isLeft else frame[12]
        shoulder = np.array(list(map(float, shoulder.strip().split(","))))

        elbow = frame[6] if isLeft else frame[13]
        elbow = np.array(list(map(float, elbow.strip().split(","))))

        # TODO: should I just calculate this once?
        # TODO:
        dv_chest_pelvis = get_dv_chest_pelvis(frame)
        dv_shoulder_elbow = np.subtract(elbow, shoulder)

        dot_product = np.dot(dv_chest_pelvis, dv_shoulder_elbow)
        norm_chest_pelvis = np.linalg.norm(dv_chest_pelvis)
        norm_shoulder_elbow = np.linalg.norm(dv_shoulder_elbow)
        
        angle = np.degrees(np.arccos(dot_product / (norm_chest_pelvis * norm_shoulder_elbow)))
        angles[frame_index] = angle

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(list(angular_velocities.values()))

    smallest_angle_index, smallest_angle = min(angles.items(), key=lambda item: item[1])
    
    biggest_angle_index, biggest_angle = max(angles.items(), key=lambda item: item[1])
    print(f"Smallest angle: {smallest_angle:.2f} in frame {smallest_angle_index} second {(smallest_angle_index / frame_rate):.2f}")
    print(f"Biggest angle: {biggest_angle:.2f} in frame {biggest_angle_index} second {(biggest_angle_index / frame_rate):.2f}")
    print("---------------------------------------")
    return angles, angular_velocities, angular_accelerations



def get_back_angle(frames, movement_start_frame, movement_end_frame):

    angles = {}

    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            print(f"Broke in frame {frame_index} of {len(frames)}")
            break
        frame = frames[frame_index]

        hip_left = np.array(list(map(float, frame[18].strip().split(","))))
        ankle_left = np.array(list(map(float, frame[20].strip().split(","))))
        hip_right = np.array(list(map(float, frame[22].strip().split(","))))
        ankle_right = np.array(list(map(float, frame[24].strip().split(","))))

        hips_midpoint = np.add(hip_left, hip_right) / 2
        ankles_midpoint = np.add(ankle_left, ankle_right) / 2

        dv_hips_ankles_mid = np.subtract(hips_midpoint, ankles_midpoint)
        dv_chest_pelvis = get_dv_chest_pelvis(frame)  # Assume this function is defined

        dv_hips_ankles_mid = dv_hips_ankles_mid / np.linalg.norm(dv_hips_ankles_mid)
        dv_chest_pelvis = dv_chest_pelvis / np.linalg.norm(dv_chest_pelvis)

        cross_product = np.cross(dv_chest_pelvis, dv_hips_ankles_mid)
        dot_product = np.dot(dv_chest_pelvis, dv_hips_ankles_mid)
        angle_radians = np.arctan2(np.linalg.norm(cross_product), dot_product)
        angle_degrees = np.degrees(angle_radians)

        # Ensure angles cover the full range
        y_component = cross_product[1]
        if y_component < 0:  # Adjust for backward bending
            angle_degrees = 360 - angle_degrees

        angles[frame_index] = angle_degrees

        # print(f"Angle (Degrees) {frame_index} {frame_index / frame_rate:.2f}s: {angle_degrees}")

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(list(angular_velocities.values()))
    
    smallest_angle_index, smallest_angle = min(angles.items(), key=lambda item: item[1])
    biggest_angle_index, biggest_angle = max(angles.items(), key=lambda item: item[1])

    percentage_depth = ((biggest_angle - smallest_angle) / biggest_angle) * 100

    print(f"Smallest angle: {smallest_angle:.2f} in frame {smallest_angle_index} second {(smallest_angle_index / frame_rate):.2f}")
    print(f"Biggest angle: {biggest_angle:.2f} in frame {biggest_angle_index} second {(biggest_angle_index / frame_rate):.2f}")
    print(f"Percentage depth: {percentage_depth:.2f}%")
    print("---------------------------------------")

    return angles, angular_velocities, angular_accelerations

def get_side_bending_angle(frames, movement_start_frame, movement_end_frame):
    angles = {}
    pain_frames, fake_frames, painless_frames, rest_frames = get_pain_and_fake_frames()

    gravity_vector = None
    mkv_name = video_name.split(".")[0] + ".mkv"
    folder_name = videos_folder.split("/")[1]
    if "All_New_Videos" == videos_folder or "1" in videos_folder or "2" in videos_folder or "5" in videos_folder:
        mkv_path = f"D:/{folder_name}/{mkv_name}"
    else:
        mkv_path = f"E:/{folder_name}/{mkv_name}"
    playback = PyK4APlayback(mkv_path)
    playback.open()
    try:
        imu_sample = playback.get_next_imu_sample()
        gravity_vector = get_gravity_vector(imu_sample, mkv_path)
        print(f"Gravity vector {gravity_vector}")
    except:
        print("IMU data is not available in this recording.")
        gravity_vector = get_dv_chest_pelvis(frames[30]) 
    
        

    # if rest_frames[0]:
    #     if 30 < rest_frames[0]:
    #         dv_rest_chest_pelvis = get_dv_chest_pelvis(frames[30]) if gravity_vector is None else gravity_vector 
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            print(f"Broke in frame {frame_index} of {len(frames)}")
            break

        frame = frames[frame_index]

        dv_chest_pelvis = get_dv_chest_pelvis(frame)
        
        # Project vectors onto the coronal plane (X, Y)
        dv_chest_pelvis_proj = dv_chest_pelvis[:2]
        gravity_vector_proj = gravity_vector[:2]

        # Normalize the projected vectors
        dv_chest_pelvis_proj = dv_chest_pelvis_proj / np.linalg.norm(dv_chest_pelvis_proj)
        gravity_vector_proj = gravity_vector_proj / np.linalg.norm(gravity_vector_proj)

        dot_product = np.dot(dv_chest_pelvis_proj, gravity_vector_proj)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_radians = np.arccos(dot_product)
        angle_degrees = np.degrees(angle_radians)

        angles[frame_index] = angle_degrees


    smallest_angle_index, smallest_angle = min(angles.items(), key=lambda item: item[1])
    biggest_angle_index, biggest_angle = max(angles.items(), key=lambda item: item[1])

    print(f"Smallest angle: {smallest_angle:.2f} in frame {smallest_angle_index} second {(smallest_angle_index / frame_rate):.2f}")
    print(f"Biggest angle: {biggest_angle:.2f} in frame {biggest_angle_index} second {(biggest_angle_index / frame_rate):.2f}")
    print("---------------------------------------")

    return angles
    


videos_folder = "./All_New_Videos_4"
videos = sorted(os.listdir(videos_folder))

for video_name in videos:
    if not video_name.endswith(".txt"):
        continue
    video_path = f"{videos_folder}/{video_name}"
    stripped_video_name = re.sub(r'v.*$', '' ,video_name)
    movement_start_frame, movement_end_frame, start_second, end_second, video_row_num, label, movement_start_frame2, movement_end_frame2 = get_movement_start_end_frames(stripped_video_name)

    frames = get_coordinates(video_path)

    if "BACK" in video_name:
        x = 1
        get_back_angle(frames, movement_start_frame, movement_end_frame)
        if movement_start_frame2 is not None:
            get_side_bending_angle(frames, movement_start_frame2, movement_end_frame2)
    elif "LEFT_SHOULDER" in video_name:
        get_shoulder_angle(frames, movement_start_frame, movement_end_frame, True)
    elif "RIGHT_SHOULDER" in video_name:
        get_shoulder_angle(frames, movement_start_frame, movement_end_frame, False)
    elif "NECK" in video_name:
        get_neck_angle(frames, movement_start_frame, movement_end_frame)
    elif "LEFT_ELBOW" in video_name:
        get_elbow_angle(frames, movement_start_frame, movement_end_frame, True)
    elif "RIGHT_ELBOW" in video_name:
        get_elbow_angle(frames, movement_start_frame, movement_end_frame, False)
    elif "ELBOWS" in video_name:
        print("Left elbow")
        get_elbow_angle(frames, movement_start_frame, movement_end_frame, True)
        print("Right elbow")
        get_elbow_angle(frames, movement_start_frame, movement_end_frame, False)
    elif "LEFT_KNEE" in video_name:
        get_knee_angle(frames, movement_start_frame, movement_end_frame, True)
    elif "RIGHT_KNEE" in video_name:
        get_knee_angle(frames, movement_start_frame, movement_end_frame, False)
    elif "KNEES" in video_name:
        print("Left knee")
        get_knee_angle(frames, movement_start_frame, movement_end_frame, True)
        print("Right knee")
        get_knee_angle(frames, movement_start_frame, movement_end_frame, False)
    elif "SHOULDERS" in video_name:
        print("Left shoulder")
        get_shoulder_angle(frames, movement_start_frame, movement_end_frame, True)
        print("Right shoulder")
        get_shoulder_angle(frames, movement_start_frame, movement_end_frame, False)
    elif "WRIST" in video_name:
        get_wrist_angle(frames, movement_start_frame, movement_end_frame)
    elif "HIP" in video_name:
        get_side_bending_angle(frames, movement_start_frame, movement_end_frame)
        

        



def create_ply_file(frame, filename='body_points3.ply'):
    # PLY header
    ply_header = '''ply
format ascii 1.0
element vertex {0}
property float x
property float y
property float z
end_header
'''.format(len(frame))

    with open(filename, 'w') as ply_file:
        ply_file.write(ply_header)
        for point in frame:
            x, y, z = map(float, point.strip().split(","))
            ply_file.write(f"{x} {y} {z}\n")  


# Create the .ply file
# create_ply_file(frame)













# import numpy as np
# from sympy.solvers import solve
# from sympy import Symbol

# def get_coordinates():
#     jointNum = 32
#     frames = []
#     frame = []
#     with open(f"REAL_BACK_EM_2024-04-23_09-20_vmaster.txt", "r") as coordinates:
#         content = coordinates.read().strip()
#         coordinate_lists = content.strip("][").split("][")

#         for coordinate_list in coordinate_lists:
#             frame.append(coordinate_list)
#             if len(frame) == jointNum:
#                 frames.append(frame)
#                 frame = []
#     return frames

# with open("./videos2/REAL_BACK_EM_2024-04-23_09-20_vmaster.txt", "r") as coordinates:
#     content = coordinates.read().strip()
#     coordinate_lists = content.strip("][").split("][")

#     frames = get_coordinates()

#     frame_index = 0  
#     joint_count = 32  

#     frame = frames[frame_index]
#     # Indexing chest, right foot, left foot from the 210th frame
#     chest = frame[2]
    

#     # Parsing joints as numpy arrays
#     chest = np.array(list(map(float, chest.strip().split(","))))

#     y = Symbol('y')
#     y = float((solve(chest[0]**2+chest[1]*y+chest[2]**2, y))[0])
    
#     ground = np.array([chest[0], y, chest[2]])

#     frame = frames[210]
#     chest = frame[2]
#     chest = np.array(list(map(float, chest.strip().split(","))))

#     # Calculate dot product between chest and pelvis
#     dot_product = np.dot(chest, ground)
#     norm_chest = np.linalg.norm(chest)
#     norm_pelvis = np.linalg.norm(ground)

#     # Calculate angle between chest and pelvis
#     angle = np.degrees(np.arccos(dot_product / (norm_chest * norm_pelvis)))
# print(f"Chest: {chest}")
# print(f"Ground: {ground}")
# print(f"Dot product: {dot_product}")
# print(f"Norm chest: {norm_chest}")
# print(f"Norm midpoint: {norm_pelvis}")
# print(f"Angle: {angle:.2f}")

# import numpy as np
# from sympy.so(lvers import solve
# from sympy ):.2fimport Symbol

# video_path = "REAL_BACK_EM_2024-04-23_09-20_vmaster.txt"

# def get_coordinates():
#     jointNum = 32
#     frames = []
#     frame = []
#     with open(video_path, "r") as coordinates:
#         content = coordinates.read().strip()
#         coordinate_lists = content.strip("][").split("][")

#         for coordinate_list in coordinate_lists:
#             frame.append(coordinate_list)
#             if len(frame) == jointNum:
#                 frames.append(frame)
#                 frame = []
#     return frames

# joints = ["PELVIS",	
# 	"SPINE_NAVAL",	
# 	"SPINE_CHEST",
# 	"NECK",
# 	"CLAVICLE_LEFT",
# 	"SHOULDER_LEFT", 
# 	"ELBOW_LEFT",
# 	"WRIST_LEFT",
# 	"HAND_LEFT",
# 	"HANDTIP_LEFT",
# 	"THUMB_LEFT",
# 	"CLAVICLE_RIGHT",
# 	"SHOULDER_RIGHT",
# 	"ELBOW_RIGHT",	
# 	"WRIST_RIGHT",	
# 	"HAND_RIGHT",
# 	"HANDTIP_RIGHT",
# 	"THUMB_RIGHT",	
# 	"HIP_LEFT",
# 	"KNEE_LEFT",	
# 	"ANKLE_LEFT",	
# 	"FOOT_LEFT",
# 	"HIP_RIGHT",
# 	"KNEE_RIGHT",
# 	"ANKLE_RIGHT",
# 	"FOOT_RIGHT",
# 	"HEAD",
# 	"NOSE",	
# 	"EYE_LEFT",
# 	"EAR_LEFT",
# 	"EYE_RIGHT",
# 	"EAR_RIGHT"]

# with open(video_path, "r") as coordinates:
#     content = coordinates.read().strip()
#     coordinate_lists = content.strip("][").split("][")

#     frames = get_coordinates()
#     frame_index = 237
#     frame = frames[frame_index]
    
#     # Read https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints#pelvis-hierarchy
#     joint_index = 0

#     chest = frame[2]
#     chest = np.array(list(map(float, chest.strip().split(","))))

#     pelvis = frame[joint_index]
#     pelvis = np.array(list(map(float, pelvis.strip().split(","))))
    

#     hip_left = frame[18]
#     hip_left = np.array(list(map(float, hip_left.strip().split(","))))

#     ankle_left = frame[20]
#     ankle_left = np.array(list(map(float, ankle_left.strip().split(","))))

#     hip_right = frame[22]
#     hip_right = np.array(list(map(float, hip_right.strip().split(","))))

#     ankle_right = frame[24]
#     ankle_right = np.array(list(map(float, ankle_right.strip().split(","))))

#     hips_midpoint = np.add(hip_left, hip_right) / 2
#     ankles_midpoint = np.add(ankle_left, ankle_right) / 2

#     dv_hips_ankles = np.subtract(hips_midpoint, ankles_midpoint)
#     dv_chest_joint = np.subtract(pelvis, chest)
                        
                
#     # Calculate dot product between chest and pelvis
#     dot_product = np.dot(chest, pelvis)
#     norm_chest = np.linalg.norm(chest)
#     norm_joint = np.linalg.norm(pelvis)
    
#     dot_product_dv = np.dot(dv_chest_joint, dv_hips_ankles)
#     norm_dv_chest_joint = np.linalg.norm(dv_chest_joint)
#     norm_dv_hips_ankles = np.linalg.norm(dv_hips_ankles)

#     # Calculate angle between chest and pelvis
#     camera_angle = np.degrees(np.arccos(dot_product / (norm_chest * norm_joint)))

#     angle = np.degrees(np.arccos(dot_product_dv / (norm_dv_chest_joint * norm_dv_hips_ankles)))
    
#     print(f"Chest: {chest}")
#     print(f"{joints[joint_index]}: {pelvis}")
#     print(f"Dot product: {dot_product}")
#     print(f"Norm chest: {norm_chest}")
#     print(f"Norm pelvis: {norm_joint}")
#     print(f"Angle relative to camera: {camera_angle:.2f}")
#     print(f"Direction vector chest (and {joints[joint_index]}: {dv_c):.2fhest_joint}")
#     print(f"Hips midpoint: {hips_midpoint}")
#     print(f"Ankles midpoint: {ankles_midpoint}")
#     print(f"Direction vector midpoint hips and ankles: {dv_hips_ankles}")
#     print(f"Actual angle: {angle:.2f}")
#     print("------------------------(---------------")

# def create_):.2fply_file(frame, filename='body_points3.ply'):
#     # PLY header
#     ply_header = '''ply
# format ascii 1.0
# element vertex {0}
# property float x
# property float y
# property float z
# end_header
# '''.format(len(frame))

#     # Write the points to the file with inverted Y-coordinate
#     with open(filename, 'w') as ply_file:
#         ply_file.write(ply_header)
#         for point in frame:
#             x, y, z = map(float, point.strip().split(","))
#             ply_file.write(f"{x} {y} {z}\n")  # Inverting Y for Blender's coordinate system






# import numpy as np
# from sympy.solvers import solve
# from sympy import Symbol
# import os
# import pandas as pd
# import re

# csv_file_path = './timings.csv'  
# movements = pd.read_csv(csv_file_path)

# frame_rate = 30


# joints = ["PELVIS",	
# 	"SPINE_NAVAL",	
# 	"SPINE_CHEST",
# 	"NECK",
# 	"CLAVICLE_LEFT",
# 	"SHOULDER_LEFT", 
# 	"ELBOW_LEFT",
# 	"WRIST_LEFT",
# 	"HAND_LEFT",
# 	"HANDTIP_LEFT",
# 	"THUMB_LEFT",
# 	"CLAVICLE_RIGHT",
# 	"SHOULDER_RIGHT",
# 	"ELBOW_RIGHT",	
# 	"WRIST_RIGHT",	
# 	"HAND_RIGHT",
# 	"HANDTIP_RIGHT",
# 	"THUMB_RIGHT",	
# 	"HIP_LEFT",
# 	"KNEE_LEFT",	
# 	"ANKLE_LEFT",	
# 	"FOOT_LEFT",
# 	"HIP_RIGHT",
# 	"KNEE_RIGHT",
# 	"ANKLE_RIGHT",
# 	"FOOT_RIGHT",
# 	"HEAD",
# 	"NOSE",	
# 	"EYE_LEFT",
# 	"EAR_LEFT",
# 	"EYE_RIGHT",
# 	"EAR_RIGHT"]

# videos = sorted(os.listdir("./videos2"))
# def get_coordinates(video_path):
#     jointNum = 32
#     frames = []
#     frame = []
#     with open(video_path, "r") as coordinates:
#         content = coordinates.read().strip()
#         coordinate_lists = content.strip("][").split("][")

#         for coordinate_list in coordinate_lists:
#             frame.append(coordinate_list)
#             if len(frame) == jointNum:
#                 frames.append(frame)
#                 frame = []
#     return frames


# def get_movement_start_end_frames(stripped_video_name):
#     movement_start_frame = None
#     movement_end_frame = None
#     start_second = None
#     end_second = None
#     video_row_num = None

#     pattern = r"^.*_(.*?)_[A-Z]{2}_.*"
#     joint_name = re.sub(pattern, r"\1", stripped_video_name)

#     row_mask = movements['Videos'] == stripped_video_name
#     if not row_mask.any():
#         print("Video not found.")
#         return None, None, None, None, None
    
#     video_row_num = row_mask.idxmax()  # first index where mask is True
#     print("File found in:", video_row_num + 2, stripped_video_name)
#     row_data = movements.loc[video_row_num]


#     for col in row_data.index[1:]:
#         # this if statement is needed because some videos have multiple types of movements
#         if "BACK" in stripped_video_name:
#             if pd.notna(row_data["BACK__BEND_FORWARD"]) and pd.notna(row_data["BACK__BEND_BACKWARD"]):
#                 start_second = list(map(int, row_data["BACK__BEND_FORWARD"].strip("()").split(",")))[0]
#                 movement_start_frame = start_second * frame_rate
#                 end_second = list(map(int, row_data["BACK__BEND_BACKWARD"].strip("()").split(",")))[1]
#                 movement_end_frame = end_second * frame_rate
#                 break
#             elif pd.notna(row_data["BACK__BEND_FORWARD_AND_BACKWARD"]):
#                 start_second, end_second = map(int, row_data["BACK__BEND_FORWARD_AND_BACKWARD"].strip("()").split(","))
#                 movement_start_frame = start_second * frame_rate
#                 movement_end_frame = end_second * frame_rate
#                 break

#         if joint_name in col and pd.notna(row_data[col]):
#             print(col)
#             start_second, end_second = map(int, row_data[col].strip("()").split(","))
#             movement_start_frame = start_second * frame_rate
#             movement_end_frame = end_second * frame_rate
#             break

#     if movement_start_frame is None or movement_end_frame is None:
#         print("Start/end not found")
#         return None, None, None, None, None
#     return movement_start_frame, movement_end_frame, start_second, end_second, video_row_num

# # Direction vector from SPINE_CHEST to PELVIS
# def get_dv_chest_pelvis(frame):
#     # Read https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints#joint-hierarchy

#     chest = frame[2]
#     chest = np.array(list(map(float, chest.strip().split(","))))

#     pelvis = frame[0]
#     pelvis = np.array(list(map(float, pelvis.strip().split(","))))
    
#     dv_chest_pelvis = np.subtract(pelvis, chest)

#     return dv_chest_pelvis

# def get_shoulder_angle(frames, movement_start_frame, movement_end_frame, isLeft):
#     angles = {}
#     for frame_index in range(movement_start_frame, movement_end_frame):
#         frame = frames[frame_index]
        
#         # Read https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints#joint-hierarchy

#         elbow = frame[6] if isLeft else frame[13]
#         elbow = np.array(list(map(float, elbow.strip().split(","))))

#         wrist = frame[7] if isLeft else frame[14]
#         wrist = np.array(list(map(float, wrist.strip().split(","))))

#         dv_chest_pelvis = get_dv_chest_pelvis(frame)
#         dv_shoulder_elbow = np.subtract(wrist, elbow)

#         dot_product = np.dot(dv_chest_pelvis, dv_shoulder_elbow)
#         norm_chest_pelvis = np.linalg.norm(dv_chest_pelvis)
#         norm_shoulder_elbow = np.linalg.norm(dv_shoulder_elbow)
        
#         angle = np.degrees(np.arccos(dot_product / (norm_chest_pelvis * norm_shoulder_elbow)))
#         angles[frame_index] = angle
#     smallest_angle_index, smallest_angle = min(angles.items(), key=lambda item: item[1])
    
#     biggest_angle_index, biggest_angle = max(angles.items(), key=lambda item: item[1])
#     print(f"Smallest angle: {smallest_angle:.2f} in frame {smallest_angle_index} second {(smallest_angle_index / frame_rate):.2f}")
#     print(f"Biggest angle: {biggest_angle:.2f} in frame {biggest_angle_index} second {(biggest_angle_index / frame_rate):.2f}")
#     print("---------------------------------------")


# def get_back_angle(frames, movement_start_frame, movement_end_frame):
#     # frame_index: angle
#     angles = {}
#     for frame_index in range(movement_start_frame, movement_end_frame):
#         frame = frames[frame_index]        

#         hip_left = frame[18]
#         hip_left = np.array(list(map(float, hip_left.strip().split(","))))

#         ankle_left = frame[20]
#         ankle_left = np.array(list(map(float, ankle_left.strip().split(","))))

#         hip_right = frame[22]
#         hip_right = np.array(list(map(float, hip_right.strip().split(","))))

#         ankle_right = frame[24]
#         ankle_right = np.array(list(map(float, ankle_right.strip().split(","))))

#         hips_midpoint = np.add(hip_left, hip_right) / 2
#         ankles_midpoint = np.add(ankle_left, ankle_right) / 2

#         # dv_hips_ankles_left = np.subtract()
#         dv_hips_ankles_mid = np.subtract(hips_midpoint, ankles_midpoint)
#         dv_chest_pelvis = get_dv_chest_pelvis(frame)
                            
        
#         dot_product_dv = np.dot(dv_chest_pelvis, dv_hips_ankles_mid)
#         norm_dv_chest_pelvis = np.linalg.norm(dv_chest_pelvis)
#         norm_dv_hips_ankles = np.linalg.norm(dv_hips_ankles_mid)

#         angle = np.degrees(np.arccos(dot_product_dv / (norm_dv_chest_pelvis * norm_dv_hips_ankles)))
#         angles[frame_index] = angle
#     smallest_angle_index, smallest_angle = min(angles.items(), key=lambda item: item[1])
    
#     biggest_angle_index, biggest_angle = max(angles.items(), key=lambda item: item[1])
#     print(f"Smallest angle: {smallest_angle:.2f} in frame {smallest_angle_index} second {(smallest_angle_index / frame_rate):.2f}")
#     print(f"Biggest angle: {biggest_angle:.2f} in frame {biggest_angle_index} second {(biggest_angle_index / frame_rate):.2f}")
#     print("---------------------------------------")

# for video_name in videos:
#     video_path = f"./videos2/{video_name}"
#     stripped_video_name = re.sub(r'v.*$', '' ,video_name)
#     movement_start_frame, movement_end_frame, start_second, end_second, video_row_num = get_movement_start_end_frames(stripped_video_name)
#     with open(video_path, "r") as coordinates:
#         content = coordinates.read().strip()
#         coordinate_lists = content.strip("][").split("][")

#         frames = get_coordinates(video_path)

#         if "BACK" in video_name:
#             x = 1
#             get_back_angle(frames, movement_start_frame, movement_end_frame)
#         elif "LEFT_SHOULDER" in video_name:
#             get_shoulder_angle(frames, movement_start_frame, movement_end_frame, True)
#         elif "RIGHT_SHOULDER" in video_name:
#             get_shoulder_angle(frames, movement_start_frame, movement_end_frame, False)
        

        



# def create_ply_file(frame, filename='body_points3.ply'):
#     # PLY header
#     ply_header = '''ply
# format ascii 1.0
# element vertex {0}
# property float x
# property float y
# property float z
# end_header
# '''.format(len(frame))

#     with open(filename, 'w') as ply_file:
#         ply_file.write(ply_header)
#         for point in frame:
#             x, y, z = map(float, point.strip().split(","))
#             ply_file.write(f"{x} {y} {z}\n")  


# # Create the .ply file
# # create_ply_file(frame)
