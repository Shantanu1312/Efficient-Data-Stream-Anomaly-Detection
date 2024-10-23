import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

# Configuration
WINDOW_SIZE = 50  # Number of previous data points to consider for Z-score calculation
Z_THRESHOLD = 2.0  # Z-score threshold for detecting anomalies
STREAM_LENGTH = 200  # Number of data points in the stream

# Simulate a data stream with regular patterns, noise, and occasional anomalies
def simulate_data_stream(length):
    data_stream = []
    for i in range(length):
        # Regular pattern: a sine wave
        value = 10 * np.sin(0.1 * i)  
        
        # Add random noise
        noise = random.uniform(-5, 5)
        value += noise
        
        # Occasionally add an anomaly
        if random.random() < 0.1:  # 10% chance of adding an anomaly
            anomaly = random.uniform(30, 50)  # Anomalies are significantly higher
            data_stream.append(anomaly)
        else:
            data_stream.append(value)
    return data_stream

# Function to detect anomalies using Z-score
def detect_anomalies(data):
    if len(data) < WINDOW_SIZE:
        return []

    mean = np.mean(data)
    std_dev = np.std(data)
    anomalies = []

    for i in range(WINDOW_SIZE, len(data)):
        z_score = (data[i] - mean) / std_dev
        if abs(z_score) > Z_THRESHOLD:
            anomalies.append(i)
    
    return anomalies

# Real-time anomaly detection and visualization
def visualize_data_stream(data_stream):
    plt.ion()  # Enable interactive mode
    plt.figure(figsize=(12, 6))
    plt.title("Real-Time Data Stream Anomaly Detection")
    plt.xlabel("Data Point Index")
    plt.ylabel("Value")
    plt.xlim(0, STREAM_LENGTH)
    plt.ylim(-20, 60)
    
    # Queue for maintaining the sliding window
    window = deque(maxlen=WINDOW_SIZE)

    for i in range(len(data_stream)):
        window.append(data_stream[i])
        anomalies = detect_anomalies(window)

        # Clear the previous plot
        plt.clf()
        plt.title("Real-Time Data Stream Anomaly Detection")
        plt.xlabel("Data Point Index")
        plt.ylabel("Value")
        plt.xlim(0, STREAM_LENGTH)
        plt.ylim(-20, 60)

        # Plot the data stream
        plt.plot(range(i + 1), data_stream[:i + 1], label='Data Stream', color='blue')

        # Plot detected anomalies
        for anomaly_index in anomalies:
            if anomaly_index == WINDOW_SIZE - 1:  # Only for the current window
                plt.scatter(anomaly_index, window[-1], color='red', label='Anomaly', s=100)

        plt.legend()
        plt.pause(0.1)  # Pause to update the plot

    plt.ioff()  # Disable interactive mode
    plt.show()

# Main function
if __name__ == "__main__":
    data_stream = simulate_data_stream(STREAM_LENGTH)
    visualize_data_stream(data_stream)
