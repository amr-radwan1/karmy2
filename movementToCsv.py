import numpy as np
import os
import pandas as pd
import re
from pyk4a import PyK4APlayback
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import OneHotEncoder
import json

csv_file_path = './timings.csv'  
movements = pd.read_csv(csv_file_path)

frame_rate = 30
time_per_frame = 1 / frame_rate

joints = {"PELVIS":0,	
    "SPINE_NAVAL": 1,	
    "HIP": 2,
    "HIPS": 2,
    "BACK": 2,
    "NECK": 3,
    "CLAVICLE_LEFT": 4,
    "SHOULDER_LEFT": 5, 
    "SHOULDERS": 5,
    "ELBOW_LEFT": 6,
    "WRIST_LEFT": 7,
    "ELBOWS": 7,
    "ELBOW": 7,
    "HAND_LEFT": 8,
    "HANDTIP_LEFT": 9,
    "THUMB_LEFT": 10,
    "CLAVICLE_RIGHT": 11,
    "SHOULDER_RIGHT": 12,
    "ELBOW_RIGHT": 13,
    "WRISTS": 14,
    "WRIST": 14,	
    "WRIST_RIGHT": 14,	
    "HAND_RIGHT": 15,
    "HANDTIP_RIGHT": 16,
    "THUMB_RIGHT": 17,	
    "HIP_LEFT": 18,
    "KNEE_LEFT": 19,	
    "ANKLE_LEFT": 20,	
    "FOOT_LEFT": 21,
    "HIP_RIGHT": 22,
    "KNEE_RIGHT": 23,
    "KNEES": 23,
    "KNEE": 23,
    "ANKLE_RIGHT": 24,
    "FOOT_RIGHT": 25,
    "HEAD": 26,
    "NOSE": 27,	
    "EYE_LEFT": 28,
    "EAR_LEFT": 29,
    "EYE_RIGHT": 30,
    "EAR_RIGHT": 31}

joints_arr = ["PELVIS",	
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

def calculate_frequency_domain_features(angle_data, sampling_rate=30):
    """
    Calculate frequency domain features from angle data
    
    Parameters:
    angle_data (list): List of angle values
    sampling_rate (int): Sampling rate in Hz, default is 30 Hz for frame rate
    
    Returns:
    dict: Dictionary containing frequency domain features
    """
    # Convert to numpy array if not already
    angles = np.array(angle_data)
    
    # Perform Fast Fourier Transform
    N = len(angles)
    yf = fft(angles)
    xf = fftfreq(N, 1/sampling_rate)[:N//2]
    amplitude_spectrum = 2.0/N * np.abs(yf[:N//2])
    
    # Find dominant frequencies (top 3)
    peaks, _ = signal.find_peaks(amplitude_spectrum)
    sorted_peaks = sorted(peaks, key=lambda i: amplitude_spectrum[i], reverse=True)
    dominant_freqs = xf[sorted_peaks[:3]] if len(sorted_peaks) >= 3 else xf[sorted_peaks]
    dominant_amps = amplitude_spectrum[sorted_peaks[:3]] if len(sorted_peaks) >= 3 else amplitude_spectrum[sorted_peaks]
    
    # Calculate spectral features
    mean_freq = np.sum(xf * amplitude_spectrum) / np.sum(amplitude_spectrum)
    median_freq = np.median(amplitude_spectrum)
    power_0_3Hz = np.sum(amplitude_spectrum[(xf >= 0) & (xf <= 3)])
    power_3_6Hz = np.sum(amplitude_spectrum[(xf > 3) & (xf <= 6)])
    power_6_10Hz = np.sum(amplitude_spectrum[(xf > 6) & (xf <= 10)])
    
    # 'mean_frequency', 'median_frequency', 'frequency_bandwidth', 'normalized_jerk'
    # return {
    #     "mean_frequency": mean_freq,
    #     "median_frequency": median_freq,
    #     "frequency_bandwidth": np.std(xf * amplitude_spectrum),
    #     "normalized_jerk": np.sum(np.diff(angles)**2) / (len(angles) * sampling_rate),
    #     "dominant_frequencies": list(dominant_freqs),
    #     "dominant_amplitudes": list(dominant_amps),
    # }
    return {
        "dominant_frequencies": list(dominant_freqs),
        "dominant_amplitudes": list(dominant_amps),
        "mean_frequency": mean_freq,
        "median_frequency": median_freq,
        "power_0_3Hz": power_0_3Hz,
        "power_3_6Hz": power_3_6Hz,
        "power_6_10Hz": power_6_10Hz,
        "frequency_bandwidth": np.std(xf * amplitude_spectrum)
    }

def calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations, frame_rate=30):
    """
    Calculate biomechanically relevant metrics from movement data
    
    Parameters:
    angles (dict): Dictionary of frame_index: angle values
    angular_velocities (dict): List of angular velocity values
    angular_accelerations (list): List of angular acceleration values
    frame_rate (int): Frame rate in Hz
    
    Returns:
    dict: Dictionary containing biomechanical metrics
    """
    angle_values = np.array(angles)
    angular_velocities_arr = np.array(angular_velocities)
    
    # Range of motion (ROM)
    rom = np.max(angle_values) - np.min(angle_values)
    
    # Movement smoothness metrics
    # Normalized jerk (lower values indicate smoother movement)
    diff_acc = np.diff(angular_accelerations)
    jerk = np.sum(diff_acc**2) / (len(diff_acc) * frame_rate)
    normalized_jerk = jerk * (np.max(angle_values) - np.min(angle_values))**2
    
    # Spectral arc length (SPARC) - simplified version
    # More negative values indicate less smooth movement
    velocity_fft = np.abs(fft(angular_velocities_arr))
    velocity_fft = velocity_fft[:len(velocity_fft)//2]
    sparc = -np.trapz(np.sqrt(1 + (velocity_fft[1:]/np.max(velocity_fft))**2))
    
    # Movement variability
    cv_angle = stats.variation(angle_values) if np.mean(angle_values) != 0 else 0
    cv_velocity = stats.variation(angular_velocities_arr) if np.mean(angular_velocities_arr) != 0 else 0
    
    # Movement speed
    mean_velocity = np.mean(np.abs(angular_velocities_arr))
    peak_velocity = np.max(np.abs(angular_velocities_arr))
    time_to_peak_velocity = np.argmax(np.abs(angular_velocities_arr)) / frame_rate
    
    # Movement acceleration
    mean_acceleration = np.mean(np.abs(angular_accelerations))
    peak_acceleration = np.max(np.abs(angular_accelerations))
    
    # Movement rhythm - coefficient of variation of timing between peaks
    peaks, _ = signal.find_peaks(angle_values)
    if len(peaks) > 1:
        interpeak_intervals = np.diff(peaks)
        rhythm_cv = np.std(interpeak_intervals) / np.mean(interpeak_intervals) if np.mean(interpeak_intervals) > 0 else 0
    else:
        rhythm_cv = 0

    # return {
    #     "spectral_arc_length": sparc,
    #     "range_of_motion": rom,
    #     "normalized_jerk": normalized_jerk,
    #     "coefficient_of_variation_angle": cv_angle,
    #     "mean_absolute_velocity": mean_velocity,
    # }
    return {
        "range_of_motion": rom,
        "normalized_jerk": normalized_jerk,
        "spectral_arc_length": sparc,
        "coefficient_of_variation_angle": cv_angle,
        "coefficient_of_variation_velocity": cv_velocity,
        "mean_absolute_velocity": mean_velocity,
        "peak_velocity": peak_velocity,
        "time_to_peak_velocity": time_to_peak_velocity,
        "mean_absolute_acceleration": mean_acceleration,
        "peak_acceleration": peak_acceleration,
        "rhythm_coefficient_of_variation": rhythm_cv
    }



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
    movement_start_end_frames = {}
    movement_start_frame = None
    movement_end_frame = None
    start_second = None
    end_second = None
    video_row_num = None

    row_mask = movements['Videos'] == stripped_video_name
    if not row_mask.any():
        print(f"{stripped_video_name} not found.")
        return None, None, None, None, None
    
    # FAKE_BACK_MB1_2024-06-04_09-07_
    video_row_num = row_mask.idxmax()  # first index where mask is True
    print("File found in:", video_row_num + 2, stripped_video_name)
    row_data = movements.loc[video_row_num]
    
    # movement_start_end_frames is a dictionary of  movement_type : [movement_start_frame, movement_end_frame)]
    if "BACK" in video_name or "SHOULDER" in video_name or "NECK" in video_name or "ELBOW" in video_name or "KNEE" in video_name or "HIP" in video_name or "WRIST" in video_name and len(movement_start_end_frames) == 0:
        for col in row_data.index[1:]:
            if joint_name in col or ("HIP" in video_name and "BACK__BEND_LEFT_AND_RIGHT" in col):
                if pd.notna(row_data[col]):
                    start_second, end_second = map(int, row_data[col].strip("()").split(","))
                    movement_start_frame = start_second * frame_rate
                    movement_end_frame = end_second * frame_rate
                    movement_start_end_frames[col] = [movement_start_frame, movement_end_frame]
    print(movement_start_end_frames)


    if movement_start_end_frames == {}:
        print(f"Movement start and end frames not found in csv file for {stripped_video_name}")
        exit()
    else:
        label = None
        print(movement_start_frame, movement_end_frame)
        pain_frames, fake_frames, painless_frames, rest_frames = get_pain_and_fake_frames()
        print(pain_frames, fake_frames)

        if pain_frames != {}:
            for start_frame, end_frame in pain_frames.items():
                for movement_start_frame, movement_end_frame in movement_start_end_frames.values():
                    if start_frame <= movement_end_frame and end_frame >= movement_start_frame:
                        label = 0
                        # print("Pain")
                        break
        if fake_frames != {}:
            for start_frame, end_frame in fake_frames.items():
                for movement_start_frame, movement_end_frame in movement_start_end_frames.values():
                    if start_frame <= movement_end_frame and end_frame >= movement_start_frame:
                        label = 1
                        # print("Not real")
                        break

        if label is None:
            if painless_frames != {}:
                for start_frame, end_frame in painless_frames.items():

                    if start_frame <= movement_end_frame and end_frame >= movement_start_frame:
                        label = 2
                        # print(f"File found in: {video_row_num + 2}, stripped_video_name: {stripped_video_name}, folder_name: {folder_name}")
                        # print(f"movement_start_frame: {movement_start_frame/30}, movement_end_frame: {movement_end_frame/30}")
                        # print(f"movement_start_end_frames: {movement_start_end_frames}")
                        # print(f"pain_frames: {pain_frames}, fake_frames: {fake_frames}, painless_frames: {painless_frames}")
                        print("Painless")
                        break

        return movement_start_frame, movement_end_frame, start_second, end_second, video_row_num, label, movement_start_end_frames

   
# Returns start_frame: end_frame dictionary for all classes in New_txt files
def get_pain_and_fake_frames():
    pain_frames = {}
    fake_frames = {}
    painless_frames = {}
    rest_frames = {}
    try:
        txt_path = rf"D:\New_txt\{stripped_video_name[:-1] if stripped_video_name[-1] == '_' else stripped_video_name}.txt"
        with open(txt_path, "r") as f:
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
        print(f"Error get_label: {e} in {stripped_video_name}")
        if "CONSISTENCY_1_REAL_BACK_BD_2024-12-11_10-07" not in video_name:
            exit()

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
    angular_velocities = []
    delta_t = 1 / frame_rate  # Time difference between frames

    previous_angle = None
    for angle in angles:
        if previous_angle is not None:
            delta_theta = angle - previous_angle
            angular_velocity = delta_theta / delta_t
            angular_velocities.append(angular_velocity)
        else: 
            angular_velocities.append(0)
        
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

def get_xyz(coords):
    return np.array(list(map(float, coords.strip().split(","))))


def get_wrist_angle(frames, movement_start_frame, movement_end_frame, isLeft):
    
    angles = []
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            break
        frame = frames[frame_index]

        left_elbow = get_xyz(frame[6] if isLeft else frame[13])
        left_wrist = get_xyz(frame[7] if isLeft else frame[14])

        dv_elbow_wrist = np.subtract(left_wrist, left_elbow)
        dv_chest_pelvis = get_dv_chest_pelvis(frame)

        dot_product = np.dot(dv_chest_pelvis, dv_elbow_wrist)
        norm_chest_pelvis = np.linalg.norm(dv_chest_pelvis)
        norm_elbow_wrist = np.linalg.norm(dv_elbow_wrist)

        angle = np.degrees(np.arccos(dot_product / (norm_chest_pelvis * norm_elbow_wrist)))

        angles.append(angle)

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)
    
    smallest_angle = min(angles)
    
    biggest_angle = max(angles)

    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics

def get_knee_angle(frames, movement_start_frame, movement_end_frame, isLeft):
    
    angles = []
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            break
        frame = frames[frame_index]
        
        hip = get_xyz(frame[18] if isLeft else frame[22])
        knee = get_xyz(frame[19] if isLeft else frame[23])
        ankle = get_xyz(frame[20] if isLeft else frame[24])

        dv_hip_knee = np.subtract(knee, hip)
        dv_knee_ankle = np.subtract(knee, ankle)

        dot_product = np.dot(dv_hip_knee, dv_knee_ankle)
        norm_hip_knee = np.linalg.norm(dv_hip_knee)
        norm_knee_ankle = np.linalg.norm(dv_knee_ankle)

        angle = np.degrees(np.arccos(dot_product / (norm_hip_knee * norm_knee_ankle)))

        angles.append(angle)

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)

    smallest_angle = min(angles)
    
    biggest_angle = max(angles)

    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics


def get_elbow_angle(frames, movement_start_frame, movement_end_frame, isLeft):
    
    angles = []
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            break
        frame = frames[frame_index]

        shoulder = get_xyz(frame[5] if isLeft else frame[12])
        elbow = get_xyz(frame[6] if isLeft else frame[13])  
        wrist = get_xyz(frame[7] if isLeft else frame[14])

        dv_shoulder_elbow = np.subtract(elbow, shoulder)
        dv_elbow_wrist = np.subtract(elbow, wrist)

        dot_product = np.dot(dv_shoulder_elbow, dv_elbow_wrist)
        norm_shoulder_elbow = np.linalg.norm(dv_shoulder_elbow)
        norm_elbow_wrist = np.linalg.norm(dv_elbow_wrist)

        angle = np.degrees(np.arccos(dot_product / (norm_shoulder_elbow * norm_elbow_wrist)))

        angles.append(angle)
        
    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)

    smallest_angle = min(angles)
    
    biggest_angle = max(angles)
    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics

def get_dv_head_nose(frame):
    head = get_xyz(frame[26])
    nose = get_xyz(frame[27])

    dv_head_nose = np.subtract(head, nose)

    return dv_head_nose

def get_neck_up_down_angle(frames, movement_start_frame, movement_end_frame):
    
    angles = []
    # reflex_angles = []
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            break
        frame = frames[frame_index]

        spine_chest = get_xyz(frame[2])
        neck = get_xyz(frame[3])

        dv_head_nose = get_dv_head_nose(frame)
        dv_neck_chest = np.subtract(neck, spine_chest)

        dot_product = np.dot(dv_neck_chest, dv_head_nose)

        angle = np.degrees(np.arccos(dot_product / (np.linalg.norm(dv_neck_chest) * np.linalg.norm(dv_head_nose))))

        angles.append(angle)

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)
    smallest_angle = min(angles)
    biggest_angle = max(angles)
    
    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics


def get_shoulder_outward_angle(frames, movement_start_frame, movement_end_frame, isLeft):
    angles = []

    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            break
        frame = frames[frame_index]
        
        # Read https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints#joint-hierarchy

        shoulder = get_xyz(frame[5] if isLeft else frame[12])
        elbow = get_xyz(frame[6] if isLeft else frame[13])

        dv_shoulder_elbow = np.subtract(elbow, shoulder)

        dv_chest_pelvis = get_dv_chest_pelvis(frame)

        dot_product = np.dot(dv_chest_pelvis, dv_shoulder_elbow)
        norm_chest_pelvis = np.linalg.norm(dv_chest_pelvis)
        norm_shoulder_elbow = np.linalg.norm(dv_shoulder_elbow)
        
        angle = np.degrees(np.arccos(dot_product / (norm_chest_pelvis * norm_shoulder_elbow)))
        angles.append(angle)

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)

    smallest_angle = min(angles)
    
    biggest_angle = max(angles)
    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics



def get_back_angle(frames, movement_start_frame, movement_end_frame):

    angles = []

    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            print(f"Broke in frame {frame_index} of {len(frames)}")
            break
        frame = frames[frame_index]

        hip_left = get_xyz(frame[18])
        ankle_left = get_xyz(frame[20])
        hip_right = get_xyz(frame[22])
        ankle_right = get_xyz(frame[24])

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

        angles.append(angle_degrees)

        # print(f"Angle (Degrees) {frame_index} {frame_index / frame_rate:.2f}s: {angle_degrees}")

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)
    
    smallest_angle = min(angles)
    biggest_angle = max(angles)

    percentage_depth = ((biggest_angle - smallest_angle) / biggest_angle) * 100


    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics

def get_side_bending_angle(frames, movement_start_frame, movement_end_frame, folder_name):
    angles = []
    # pain_frames, fake_frames, painless_frames, rest_frames = get_pain_and_fake_frames()

    gravity_vector = None
    mkv_name = video_name.split(".")[0] + ".mkv"
    # path depending on which disk the folder is in 
    if "All_New_Videos" == folder_name or "1" in folder_name or "2" in folder_name or "5" in folder_name:
        mkv_path = f"D:/{folder_name}/{mkv_name}"
    else:
        mkv_path = f"E:/{folder_name}/{mkv_name}"
    # playback = PyK4APlayback(mkv_path)
    # playback.open()
    # try:
    #     imu_sample = playback.get_next_imu_sample()
    #     gravity_vector = get_gravity_vector(imu_sample, mkv_path)
    #     print(f"Gravity vector {gravity_vector}")
    # except:
    #     print("IMU data is not available in this recording.")
    #     # gets patient's rest vector
    #     gravity_vector = get_dv_chest_pelvis(frames[30]) 
    gravity_vector = get_dv_chest_pelvis(frames[30]) 

        

    # if rest_frames[0]:
    #     if 30 < rest_frames[0]:
    #         dv_rest_chest_pelvis = get_dv_chest_pelvis(frames[30]) if gravity_vector is None else gravity_vector 
    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
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

        angles.append(angle_degrees)


    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)

    smallest_angle = min(angles)
    
    biggest_angle = max(angles)
    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics


def get_shoulder_front_angle(frames, movement_start_frame, movement_end_frame, isLeft):

    shoulder = get_xyz(frames[30][5] if isLeft else frames[30][12])
    elbow = get_xyz(frames[30][6] if isLeft else frames[30][13])

    dv_shoulder_elbow_rest = np.subtract(elbow, shoulder)

    angles = []

    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            break
        frame = frames[frame_index]
        
        # Read https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints#joint-hierarchy

        shoulder = get_xyz(frame[5] if isLeft else frame[12])
        elbow = get_xyz(frame[6] if isLeft else frame[13])
        dv_shoulder_elbow = np.subtract(elbow, shoulder)

        dot_product = np.dot(dv_shoulder_elbow_rest, dv_shoulder_elbow)
        norm_shoulder_elbow_rest = np.linalg.norm(dv_shoulder_elbow_rest)
        norm_shoulder_elbow = np.linalg.norm(dv_shoulder_elbow)

        dot_product_normalized = max(min(dot_product / (norm_shoulder_elbow_rest * norm_shoulder_elbow), 1.0), -1.0)

        angle = np.degrees(np.arccos(dot_product_normalized))
        angles.append(angle)

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)

    smallest_angle = min(angles)
    
    biggest_angle = max(angles)
    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics

# Finds angle between ears relative to the x axis and
# subtracts the shoulder angle in case the person moves their whole body rather than just tilting
def get_head_tilt_angle(frames, movement_start_frame, movement_end_frame):
    angles = []
    stable_angles = []

    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            break
            x
        frame = frames[frame_index]
        
        left_ear = np.array(list(map(float, frame[29].strip().split(","))))[:2]
        right_ear = np.array(list(map(float, frame[30].strip().split(","))))[:2]
        left_shoulder = np.array(list(map(float, frame[5].strip().split(","))))[:2]
        right_shoulder = np.array(list(map(float, frame[12].strip().split(","))))[:2]
        
        ear_angle = np.degrees(np.arctan2(
            right_ear[1] - left_ear[1],
            right_ear[0] - left_ear[0]
        ))
        shoulder_angle = np.degrees(np.arctan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        ))
        
        diff = ear_angle - shoulder_angle
        diff = ((diff + 180) % 360) - 180  # normalizes the angle difference to [-180, 180]

        stable_angles.append(diff)
        
        if len(stable_angles) > 5:
            median_angle = np.median(stable_angles)
            std_dev = np.std(stable_angles)
            
            filtered_angles = [
                angle for angle in stable_angles 
                if abs(angle - median_angle) <= 1.5 * std_dev
            ]
            
            tilt_angle = np.median(filtered_angles)
        else:
            tilt_angle = diff

        angles.append(tilt_angle)

        if len(stable_angles) > 10:
            stable_angles.pop(0)

    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)

    smallest_angle = min(angles)
    
    biggest_angle = max(angles)
    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics

def get_head_yaw_angle(frames, movement_start_frame, movement_end_frame):
    angles = []

    for frame_index in range(movement_start_frame, movement_end_frame):
        if frame_index > len(frames) - 1:
            break
            
        frame = frames[frame_index]

        nose = np.array(list(map(float, frame[27].strip().split(","))))  # Nose point
        neck = np.array(list(map(float, frame[3].strip().split(","))))  # Neck point
        left_shoulder = np.array(list(map(float, frame[5].strip().split(","))))
        right_shoulder = np.array(list(map(float, frame[12].strip().split(","))))
        

        shoulder_vector = right_shoulder - left_shoulder
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        vertical_vector = mid_shoulder - neck
        

        body_normal = np.cross(shoulder_vector[:3], vertical_vector[:3])
        body_normal = body_normal / np.linalg.norm(body_normal)  
        

        nose_projected = nose.copy()
        nose_projected[2] -= 10  
        head_vector = nose_projected - neck
        head_vector = head_vector / np.linalg.norm(head_vector)
        
        body_normal_xz = np.array([body_normal[0], body_normal[2]])
        head_vector_xz = np.array([head_vector[0], head_vector[2]])
        
        body_normal_xz = body_normal_xz / np.linalg.norm(body_normal_xz)
        head_vector_xz = head_vector_xz / np.linalg.norm(head_vector_xz)
        
        yaw_angle = np.degrees(np.arccos(np.clip(np.dot(body_normal_xz, head_vector_xz), -1.0, 1.0)))
        
        cross_product = np.cross(body_normal_xz, head_vector_xz)
        if cross_product < 0:
            yaw_angle = -yaw_angle
            
        angles.append(yaw_angle)
    
    angular_velocities = calculate_angular_velocity(angles)
    angular_accelerations = calculate_angular_acceleration(angular_velocities)
    frequency_domain = calculate_frequency_domain_features(angles)
    biomechanical_metrics = calculate_biomechanical_metrics(angles, angular_velocities, angular_accelerations)

    smallest_angle = min(angles)
    
    biggest_angle = max(angles)
    return angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics


# https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/body-joints
# Takes in the frame and index of joint which is found using the joints dictionary
# For most pains, the bone for the pain is joint_index and joint_index - 1
def get_direction_vector(frame, joint_index):
    # I'm also using joint index 7 for elbows as it makes it easy to automate with the rest of the joints
    # because best direction vector for elbows is wrist with elbow and wrist index is 7 and elbow index is 6 
    # thus for wrists need to check if joint_index is 7 and joint_name is "WRISTS"
    if joint_index == 7 and joint_name in "WRISTS":
        # Wrist with handtip for wrist videos
        joint1 = np.array(list(map(float, frame[joint_index].strip().split(","))))
        joint2 = np.array(list(map(float, frame[joint_index+2].strip().split(","))))

        return np.subtract(joint2, joint1)
    # Joint index 3 is NECK and best direction vector for neck speed found is neck with head (26)
    if joint_index != 3:
        # Shoulder index - 1 would be clavicle which isnt the ideal direction vector so adding one would 
        # make the direction ector elbow with shoulder
        # 5 is shoulder left 12 is shoulder right
        if joint_index == 5 or joint_index == 12:
            joint_index += 1
        # Example: video is elbow pain then index would be 6. 6-1 is shoulder index so direction vector would be elbow and shoulder
        # AB = OB - OA
        joint1 = np.array(list(map(float, frame[joint_index-1].strip().split(","))))
        joint2 = np.array(list(map(float, frame[joint_index].strip().split(","))))

        return np.subtract(joint2, joint1)
    else:
        # Direction vector for neck videos is neck with head
        joint1 = np.array(list(map(float, frame[joint_index].strip().split(","))))
        joint2 = np.array(list(map(float, frame[26].strip().split(","))))

        return np.subtract(joint2, joint1)



count1 = 0
def export_movement_to_csv(video_name, joint, movement_name, angles, angular_velocities, angular_accelerations, label, 
                            frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file):
    global count1

    if not angles:  
        print(f"Angles list is empty for {video_name}, {joint}, {movement_name}")
        return

    scalar_data = {
        'num_frames': len(angles),
        'min_angle': min(angles),
        'max_angle': max(angles),
        'mean_angle': np.mean(angles),
        'std_angle': np.std(angles),
    }

    converted_joint = joint_mapping.get(joint, joint) 

    angle_features = {
        'angle': json.dumps(angles),
        'angular_velocity': json.dumps(angular_velocities),
        'angular_acceleration': json.dumps(list(angular_accelerations)),
    }

    movement_features = {
        'distance': json.dumps(distances[movement_name]),
        'velocity': json.dumps(velocities[movement_name]),
        'acceleration': json.dumps(accelerations[movement_name]),
    }


    aggregated_features = {
        'joint': converted_joint,
        'video_name': video_name,
        'movement': movement_name,
        'label': label,
    }
    aggregated_features.update(angle_features)
    aggregated_features.update(movement_features)
    aggregated_features.update(scalar_data)
    aggregated_features.update(frequency_domain)
    aggregated_features.update(biomechanical_metrics)


    count1 += 1

    agg_df = pd.DataFrame([aggregated_features])
    for j in all_joints:
        agg_df[f"joint_{j}"] = (agg_df['joint'] == j).astype(int)
    
    for m in all_movements:
        agg_df[f"movement_{m}"] = (agg_df['movement'] == m).astype(int)

    if not os.path.exists(master_file):
        agg_df.to_csv(master_file, index=False)
    else:
        agg_df.to_csv(master_file, mode='a', header=False, index=False)

    



master_file = "movement_csv/neck_movements.csv"
if os.path.exists(master_file):
    os.remove(master_file)

# all_joints = ['HIPS', 'HIP', 'KNEES', 'BACK', 'SHOULDERS', 'NECK', 'ELBOW', 'SHOULDER', 'WRISTS', 'KNEE', 'WRIST', 'ELBOWS']
all_joints = ['HIPS', 'KNEES', 'BACK', 'SHOULDERS', 'NECK', 'ELBOWS', 'WRISTS']
joint_mapping = {
    'HIP': 'HIPS',
    'KNEE': 'KNEES',
    'ELBOW': 'ELBOWS',
    'WRIST': 'WRISTS',
    'SHOULDER': 'SHOULDERS'
}

all_movements = [
    "BACK__BEND_FORWARD_AND_BACKWARD",
    "BACK__BEND_LEFT_AND_RIGHT",
    "SHOULDER__RAISE_ARMS_OUTWARD__LEFT",
    "SHOULDER__RAISE_ARMS_OUTWARD__RIGHT",
    "SHOULDER__RAISE_ARMS_IN_FRONT__LEFT",
    "SHOULDER__RAISE_ARMS_IN_FRONT__RIGHT",
    "NECK__LOOK_LEFT_AND_RIGHT",
    "NECK__LOOK_UP_AND_DOWN",
    "NECK__TILT_HEAD_TO_LEFT_AND_RIGHT",
    "ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__LEFT",
    "ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__RIGHT",
    "KNEES__SQUAT_DOWN",
    "WRISTS__BEND_LEFT_AND_RIGHT__LEFT",
    "WRISTS__BEND_LEFT_AND_RIGHT__RIGHT",
]

fake_count = 0
real_count = 0
count_painless = 0
missing_ang = []
for i in range(7):  
    folder_name = f"All_New_Videos" if i == 0 else f"All_New_Videos_{i}"
    if i == 6:
        folder_name = "New_Uploads"
    count = 0


    if os.path.exists(folder_name) and os.path.isdir(folder_name):  #
        print(f"Processing folder: {folder_name}") 
        
        for video_name in os.listdir(folder_name):     # angle1 missing                                                 # not fully loaded
            if not video_name.endswith(".txt") or "CONSISTENCY_2_LEFT_SHOULDER_RK_2024-04-16_09-31_" in video_name or "CONSISTENCY_2_REAL_BACK_RK_2024-04-23_09-20_" in video_name or "REAL_BACK_MC_2024-02-16_16-25_" in video_name:
                continue
            if "FAKE_RIGHT_WRIST_NB_2024-07-02_09-50_" in video_name or "ARM" in video_name:
                continue

            if "REAL" not in video_name and "FAKE" not in video_name and "RAKE" not in video_name:
                continue

            if "NECK" not in video_name:
                continue

            if "REAL" in video_name:
                real_count += 1
            elif "FAKE" in video_name or "RAKE" in video_name:
                fake_count += 1
            count += 1


            if "CONSISTENCY_2_REAL_RIGHT_SHOULDER_JP_2024-03-28_09-37" in video_name:
                continue

            video_path = f"{folder_name}/{video_name}"

            stripped_video_name = re.sub(r'v.*$', '' ,video_name) if folder_name != "New_Uploads" else re.sub(r'_v.*$', '' ,video_name)

            pattern = r"^.*_(.*?)_[A-Z]{2}\d?_.*"
            joint_name = re.sub(pattern, r"\1", stripped_video_name)

            
            movement_start_frame, movement_end_frame, start_second, end_second, video_row_num, label, movement_start_end_frames = get_movement_start_end_frames(stripped_video_name)

            if label is None:
                if "CONSISTENCY_1_REAL_BACK_BD_2024-12-11_10-07" in video_name:
                    label = 0
                else:
                    continue
            elif label == 2:
                count_painless += 1
                if "REAL" in video_name:
                    label = 0
                elif "FAKE" in video_name:
                    label = 1
                continue

            # elif label == 2:
            #     if "REAL" in video_name:
            #         label = 0
            #     elif "FAKE" in video_name:
            #         label = 1

            # if label != 2:
            #     continue

            frames = get_coordinates(video_path)

            if "LEFT_SHOULDER" in stripped_video_name:
                joint_index = 5
            elif "RIGHT_SHOULDER" in stripped_video_name:
                joint_index = 12
            elif "SHOULDERS" in stripped_video_name:
                joint_index = 5
            elif "ELBOWS" in stripped_video_name:
                joint_index = 6
            elif "KNEES" in stripped_video_name:
                joint_index = 19
            elif "NECK" in stripped_video_name:
                # I think this would make the speed wrong
                joint_index = 26
            else:
                joint_index = joints[joint_name]

            velocities = {}
            distances = {}
            accelerations = {}

            for movement in movement_start_end_frames:
                stripped_movement = movement.replace("BOTH", "")

                joint_index2 = None
                if "SHOULDERS__RAISE_ARMS_IN_FRONT__BOTH" in movement or "SHOULDERS__RAISE_ARMS_OUTWARD__BOTH" in movement:
                    joint_index2 = 12
                elif "ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__BOTH" in movement:
                    joint_index2 = 13
                elif "WRISTS__BEND_LEFT_AND_RIGHT__BOTH" in movement:
                    joint_index2 = 14
                
                if "BOTH" in movement:
                    distances[f"{stripped_movement}LEFT"] = []
                    distances[f"{stripped_movement}RIGHT"] = []
                    velocities[f"{stripped_movement}LEFT"] = []
                    velocities[f"{stripped_movement}RIGHT"] = []
                else:
                    velocities[movement] = []
                    distances[movement] = []

                for frame_index in range(movement_start_end_frames[movement][0], movement_start_end_frames[movement][1]):
                    if frame_index > len(frames) - 2:
                        print(f"Broke in frame {frame_index} of {movement_start_end_frames[movement][1]}")
                        break
                    
                    dv = get_direction_vector(frames[frame_index], joint_index)
                    dv2 = get_direction_vector(frames[frame_index + 1], joint_index)

                    distance = np.linalg.norm(np.subtract(dv2, dv))
                    velocity = (distance / time_per_frame)

                    if not joint_index2:
                        distances[movement].append(distance)
                        velocities[movement].append(velocity)
                    else:
                        dv3 = get_direction_vector(frames[frame_index], joint_index2)
                        dv4 = get_direction_vector(frames[frame_index + 1], joint_index2)

                        distance2 = np.linalg.norm(np.subtract(dv4, dv3))
                        velocity2 = (distance2 / time_per_frame)
                            
                        distances[f"{stripped_movement}LEFT"].append(distance)
                        distances[f"{stripped_movement}RIGHT"].append(distance2)
                        velocities[f"{stripped_movement}LEFT"].append(velocity)
                        velocities[f"{stripped_movement}RIGHT"].append(velocity2)

            for movement in distances:
                accelerations[movement] = []
                for i in range(1, len(velocities[movement])):
                    acceleration = (velocities[movement][i] - velocities[movement][i - 1]) / time_per_frame
                    accelerations[movement].append(acceleration)


            angles, angular_velocities, angles2, angular_velocities2, angles3, angular_velocities3, angles4, angular_velocities4 = [], [], [], [], [], [], [], []
            angular_accelerations, angular_accelerations2, angular_accelerations3, angular_accelerations4 = np.array([]), np.array([]), np.array([]), np.array([])
            frequency_domain, frequency_domain2, frequency_domain3, frequency_domain4 = {}, {}, {}, {}
            biomechanical_metrics, biomechanical_metrics2, biomechanical_metrics3, biomechanical_metrics4 = {}, {}, {}, {}

            if "BACK" in video_name:
                if movement_start_end_frames.get("BACK__BEND_FORWARD_AND_BACKWARD", None) is not None:
                    angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_back_angle(frames, movement_start_end_frames["BACK__BEND_FORWARD_AND_BACKWARD"][0], movement_start_end_frames["BACK__BEND_FORWARD_AND_BACKWARD"][1])
                    export_movement_to_csv(video_name, joint_name, "BACK__BEND_FORWARD_AND_BACKWARD", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
                if movement_start_end_frames.get("BACK__BEND_LEFT_AND_RIGHT", None) is not None:
                    angles2, angular_velocities2, angular_accelerations2, frequency_domain2, biomechanical_metrics2 = get_side_bending_angle(frames, movement_start_end_frames["BACK__BEND_LEFT_AND_RIGHT"][0], movement_start_end_frames["BACK__BEND_LEFT_AND_RIGHT"][1], folder_name)
                    export_movement_to_csv(video_name, joint_name, "BACK__BEND_LEFT_AND_RIGHT", angles2, angular_velocities2, angular_accelerations2, label, frequency_domain2, biomechanical_metrics2, distances, velocities, accelerations, master_file)
            elif "LEFT_SHOULDER" in video_name:
                if movement_start_end_frames.get("SHOULDER__RAISE_ARMS_OUTWARD__LEFT", None) is not None:
                    angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_shoulder_outward_angle(frames, movement_start_end_frames["SHOULDER__RAISE_ARMS_OUTWARD__LEFT"][0], movement_start_end_frames["SHOULDER__RAISE_ARMS_OUTWARD__LEFT"][1], True)
                    export_movement_to_csv(video_name, joint_name, "SHOULDER__RAISE_ARMS_OUTWARD__LEFT", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
                if movement_start_end_frames.get("SHOULDER__RAISE_ARMS_IN_FRONT__LEFT", None) is not None:
                    angles2, angular_velocities2, angular_accelerations2, frequency_domain2, biomechanical_metrics2 = get_shoulder_front_angle(frames, movement_start_end_frames["SHOULDER__RAISE_ARMS_IN_FRONT__LEFT"][0], movement_start_end_frames["SHOULDER__RAISE_ARMS_IN_FRONT__LEFT"][1], True)
                    export_movement_to_csv(video_name, joint_name, "SHOULDER__RAISE_ARMS_IN_FRONT__LEFT", angles2, angular_velocities2, angular_accelerations2, label, frequency_domain2, biomechanical_metrics2, distances, velocities, accelerations, master_file)
            elif "RIGHT_SHOULDER" in video_name:
                if movement_start_end_frames.get("SHOULDER__RAISE_ARMS_OUTWARD__RIGHT", None) is not None:
                    angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_shoulder_outward_angle(frames, movement_start_end_frames["SHOULDER__RAISE_ARMS_OUTWARD__RIGHT"][0], movement_start_end_frames["SHOULDER__RAISE_ARMS_OUTWARD__RIGHT"][1], False)
                    export_movement_to_csv(video_name, joint_name, "SHOULDER__RAISE_ARMS_OUTWARD__RIGHT", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
                if movement_start_end_frames.get("SHOULDER__RAISE_ARMS_IN_FRONT__RIGHT", None) is not None:
                    angles2, angular_velocities2, angular_accelerations2, frequency_domain2, biomechanical_metrics2 = get_shoulder_front_angle(frames, movement_start_end_frames["SHOULDER__RAISE_ARMS_IN_FRONT__RIGHT"][0], movement_start_end_frames["SHOULDER__RAISE_ARMS_IN_FRONT__RIGHT"][1], False)
                    export_movement_to_csv(video_name, joint_name, "SHOULDER__RAISE_ARMS_IN_FRONT__RIGHT", angles2, angular_velocities2, angular_accelerations2, label, frequency_domain2, biomechanical_metrics2, distances, velocities, accelerations, master_file)
            elif "NECK" in video_name:
                if movement_start_end_frames.get("NECK__LOOK_LEFT_AND_RIGHT", None) is not None:
                    angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_head_yaw_angle(frames, movement_start_end_frames["NECK__LOOK_LEFT_AND_RIGHT"][0], movement_start_end_frames["NECK__LOOK_LEFT_AND_RIGHT"][1])
                    export_movement_to_csv(video_name, joint_name, "NECK__LOOK_LEFT_AND_RIGHT", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
                if movement_start_end_frames.get("NECK__LOOK_UP_AND_DOWN", None) is not None:
                    angles2, angular_velocities2, angular_accelerations2, frequency_domain2, biomechanical_metrics2 = get_neck_up_down_angle(frames, movement_start_end_frames["NECK__LOOK_UP_AND_DOWN"][0], movement_start_end_frames["NECK__LOOK_UP_AND_DOWN"][1])
                    export_movement_to_csv(video_name, joint_name, "NECK__LOOK_UP_AND_DOWN", angles2, angular_velocities2, angular_accelerations2, label, frequency_domain2, biomechanical_metrics2, distances, velocities, accelerations, master_file)
                if movement_start_end_frames.get("NECK__TILT_HEAD_TO_LEFT_AND_RIGHT", None) is not None:
                    angles3, angular_velocities3, angular_accelerations3, frequency_domain3, biomechanical_metrics3 = get_head_tilt_angle(frames, movement_start_end_frames["NECK__TILT_HEAD_TO_LEFT_AND_RIGHT"][0], movement_start_end_frames["NECK__TILT_HEAD_TO_LEFT_AND_RIGHT"][1])
                    export_movement_to_csv(video_name, joint_name, "NECK__TILT_HEAD_TO_LEFT_AND_RIGHT", angles3, angular_velocities3, angular_accelerations3, label, frequency_domain3, biomechanical_metrics3, distances, velocities, accelerations, master_file)
            elif "LEFT_ELBOW" in video_name:
                angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_elbow_angle(frames, movement_start_frame, movement_end_frame, True)
                export_movement_to_csv(video_name, joint_name, "ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__LEFT", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
            elif "ELBOWS" in video_name:
                if movement_start_end_frames.get("ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__BOTH", None) is not None:
                    angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_elbow_angle(frames, movement_start_end_frames["ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__BOTH"][0], movement_start_end_frames["ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__BOTH"][1], True)
                    angles2, angular_velocities2, angular_accelerations2, frequency_domain2, biomechanical_metrics2 = get_elbow_angle(frames, movement_start_end_frames["ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__BOTH"][0], movement_start_end_frames["ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__BOTH"][1], False)
                    export_movement_to_csv(video_name, joint_name, "ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__LEFT", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
                    export_movement_to_csv(video_name, joint_name, "ELBOW__RAISE_ARMS_IN_FRONT_AND_FLEX_ELBOWS__RIGHT", angles2, angular_velocities2, angular_accelerations2, label, frequency_domain2, biomechanical_metrics2, distances, velocities, accelerations, master_file)
            elif "LEFT_KNEE" in video_name:
                angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_knee_angle(frames, movement_start_end_frames["KNEES__SQUAT_DOWN"][0], movement_start_end_frames["KNEES__SQUAT_DOWN"][1], True)
                export_movement_to_csv(video_name, joint_name, "KNEES__SQUAT_DOWN", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
            elif "RIGHT_KNEE" in video_name:
                angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_knee_angle(frames, movement_start_end_frames["KNEES__SQUAT_DOWN"][0], movement_start_end_frames["KNEES__SQUAT_DOWN"][1], False)
                export_movement_to_csv(video_name, joint_name, "KNEES__SQUAT_DOWN", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
            elif "KNEES" in video_name:
                angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_knee_angle(frames, movement_start_end_frames["KNEES__SQUAT_DOWN"][0], movement_start_end_frames["KNEES__SQUAT_DOWN"][1], True)
                angles2, angular_velocities2, angular_accelerations2, frequency_domain2, biomechanical_metrics2 = get_knee_angle(frames, movement_start_end_frames["KNEES__SQUAT_DOWN"][0], movement_start_end_frames["KNEES__SQUAT_DOWN"][1], False)
                export_movement_to_csv(video_name, joint_name, "KNEES__SQUAT_DOWN", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
                export_movement_to_csv(video_name, joint_name, "KNEES__SQUAT_DOWN", angles2, angular_velocities2, angular_accelerations2, label, frequency_domain2, biomechanical_metrics2, distances, velocities, accelerations, master_file)
            elif "SHOULDERS" in video_name:
                if movement_start_end_frames.get("SHOULDERS__RAISE_ARMS_OUTWARD__BOTH", None) is not None:
                    angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_shoulder_outward_angle(frames, movement_start_end_frames["SHOULDERS__RAISE_ARMS_OUTWARD__BOTH"][0], movement_start_end_frames["SHOULDERS__RAISE_ARMS_OUTWARD__BOTH"][1], True)
                    angles3, angular_velocities3, angular_accelerations3, frequency_domain3, biomechanical_metrics3 = get_shoulder_outward_angle(frames, movement_start_end_frames["SHOULDERS__RAISE_ARMS_OUTWARD__BOTH"][0], movement_start_end_frames["SHOULDERS__RAISE_ARMS_OUTWARD__BOTH"][1], False)
                    export_movement_to_csv(video_name, joint_name, "SHOULDERS__RAISE_ARMS_OUTWARD__LEFT", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
                    export_movement_to_csv(video_name, joint_name, "SHOULDERS__RAISE_ARMS_OUTWARD__RIGHT", angles3, angular_velocities3, angular_accelerations3, label, frequency_domain3, biomechanical_metrics3, distances, velocities, accelerations, master_file)
                if movement_start_end_frames.get("SHOULDERS__RAISE_ARMS_IN_FRONT__BOTH", None) is not None:
                    angles2, angular_velocities2, angular_accelerations2, frequency_domain2, biomechanical_metrics2 = get_shoulder_front_angle(frames, movement_start_end_frames["SHOULDERS__RAISE_ARMS_IN_FRONT__BOTH"][0], movement_start_end_frames["SHOULDERS__RAISE_ARMS_IN_FRONT__BOTH"][1], True)
                    angles4, angular_velocities4, angular_accelerations4, frequency_domain4, biomechanical_metrics4 = get_shoulder_front_angle(frames, movement_start_end_frames["SHOULDERS__RAISE_ARMS_IN_FRONT__BOTH"][0], movement_start_end_frames["SHOULDERS__RAISE_ARMS_IN_FRONT__BOTH"][1], False)
                    export_movement_to_csv(video_name, joint_name, "SHOULDERS__RAISE_ARMS_IN_FRONT__LEFT", angles2, angular_velocities2, angular_accelerations2, label, frequency_domain2, biomechanical_metrics2, distances, velocities, accelerations, master_file)
                    export_movement_to_csv(video_name, joint_name, "SHOULDERS__RAISE_ARMS_IN_FRONT__RIGHT", angles4, angular_velocities4, angular_accelerations4, label, frequency_domain4, biomechanical_metrics4, distances, velocities, accelerations, master_file)
            elif "WRISTS" in video_name:
                angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_wrist_angle(frames, movement_start_end_frames["WRISTS__BEND_LEFT_AND_RIGHT__BOTH"][0], movement_start_end_frames["WRISTS__BEND_LEFT_AND_RIGHT__BOTH"][1], True)
                angles2, angular_velocities2, angular_accelerations2, frequency_domain2, biomechanical_metrics2 = get_wrist_angle(frames, movement_start_end_frames["WRISTS__BEND_LEFT_AND_RIGHT__BOTH"][0], movement_start_end_frames["WRISTS__BEND_LEFT_AND_RIGHT__BOTH"][1], False)
                export_movement_to_csv(video_name, joint_name, "WRISTS__BEND_LEFT_AND_RIGHT__LEFT", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
                export_movement_to_csv(video_name, joint_name, "WRISTS__BEND_LEFT_AND_RIGHT__RIGHT", angles2, angular_velocities2, angular_accelerations2, label, frequency_domain2, biomechanical_metrics2, distances, velocities, accelerations, master_file)
            elif "LEFT_WRIST" in video_name:
                angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_wrist_angle(frames, movement_start_end_frames["WRIST__BEND_LEFT_AND_RIGHT__LEFT"][0], movement_start_end_frames["WRIST__BEND_LEFT_AND_RIGHT__LEFT"][1], True)
                export_movement_to_csv(video_name, joint_name, "WRIST__BEND_LEFT_AND_RIGHT__LEFT", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
            elif "HIP" in video_name:
                angles, angular_velocities, angular_accelerations, frequency_domain, biomechanical_metrics = get_side_bending_angle(frames, movement_start_end_frames["BACK__BEND_LEFT_AND_RIGHT"][0], movement_start_end_frames["BACK__BEND_LEFT_AND_RIGHT"][1], folder_name)
                export_movement_to_csv(video_name, joint_name, "BACK__BEND_LEFT_AND_RIGHT", angles, angular_velocities, angular_accelerations, label, frequency_domain, biomechanical_metrics, distances, velocities, accelerations, master_file)
            
    print(f"Exported aggregated features to: ./{master_file}")
    print(f"Count: {count}")
print(f"Count1: {count1}")
print(f"Missing angles: {missing_ang}")
print(f"Count painless: {count_painless}")
print(f"Fake count: {fake_count}")
print(f"Real count: {real_count}")
