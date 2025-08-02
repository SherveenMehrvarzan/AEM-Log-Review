# Sherveen Mehrvarzan
# AEM Log Output

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# USER EDITABLE VARIABLES

# Get the input file (ideally telem data)
aemLog = 'aemLogSkidpad.csv'
freqData = 200.0  # Frequency of data collection in Hz

# Vehicle Varaibles
wheelRadius = 0.206 # meters
vehicleMass = 210 # kg

# Setup Variables
var_mapping = {
    "IVT_current": "IVTcurrent",
    "AMKFR_actualVelocity": "FRrpm",
    "AMKFL_actualVelocity": "FLrpm", 
    "AMKRL_actualVelocity": "RLrpm",
    "AMKRR_actualVelocity": "RRrpm",
    "AMKFR_actualTorque": "torqueFR",
    "AMKFL_actualTorque": "torqueFL",
    "AMKRL_actualTorque": "torqueRL", 
    "AMKRR_actualTorque": "torqueRR",
    "IVT_voltage1": "IVTvoltage",
    "IMU_lin_body_acc_z": "accelZ",
    "IMU_lin_body_acc_y": "accelY", 
    "IMU_lin_body_acc_x": "accelX",
    "IMU_pos_lat": "latitude",
    "IMU_pos_lon": "longitude",
    "ECU_tcSteeringDegrees": "steeringAngle",
    "IMU_vel_y": "velocityY",
    "IMU_pitch_deg": "pitch",
    "IMU_yaw_deg": "yaw",
    "IMU_roll_deg": "roll",
}

def main():
    try:
        df = pd.read_csv(aemLog)
        print("CSV loaded successfully!")
        # Get the first line (fileVariables) and create variables for each column
        fileVariables = list(df.columns)
        # Dynamically create variables for each column name
        for col in fileVariables:
            globals()[col] = col
    except FileNotFoundError:
        print(f"Error: The file '{aemLog}' was not found.")
    except pd.errors.ParserError:
        print("Error: Failed to parse the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return df, fileVariables

def assignVaraibles(fileVariables, df):
    # Alternative approach using a dictionary
    finalData = {}
    for df_col, var_name in var_mapping.items():
        if df_col in fileVariables:
            finalData[var_name] = df[df_col].values
        else:
            print(f"Column '{df_col}' not found in CSV file")

    # Test print to verify variable assignments
    print("\nVariable value verification:")
    for var_name in var_mapping.values():
        if var_name in globals():
            print(f"{var_name}: First few values = {globals()[var_name][:5]}")

    return finalData

def gLoadGraph(finalData):
    # Create acceleration scatter plot
    plt.figure(figsize=(10, 10))  # Square figure for better visualization
    
    # Convert accelerations from m/s² to g's (1g = 9.81 m/s²)
    accel_x = finalData["accelX"]/9.81  # Lateral acceleration
    accel_y = finalData["accelY"]/-9.81  # Longitudinal acceleration
    
    # Calculate total g-load using Milliken's equation
    total_g = np.sqrt(accel_x**2 + accel_y**2)
    max_g = np.max(total_g)
    
    # Create scatter plot
    plt.scatter(accel_x, accel_y, 
            alpha=0.5,  
            c=total_g,  # Color based on total g-load
            cmap='viridis',  # Use viridis colormap
            s=1)        

    plt.grid(True)
    plt.xlabel('Lateral Gs')
    plt.ylabel('Longitudinal Gs')
    plt.title(f'Vehicle G Profile (Max Combined G: {max_g:.2f}g)')
    
    # Add colorbar
    plt.colorbar(label='Combined G-Load')

    # Add center lines
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)

    # Add theoretical grip limit ellipse (using Milliken's typical grip circle)
    # Typically 1.5-1.6g for racing tires on dry asphalt
    grip_limit = 1.6
    ellipse = plt.matplotlib.patches.Ellipse((0, 0), 
                                           width=grip_limit*2,  # Lateral grip limit
                                           height=grip_limit*2, # Longitudinal grip limit
                                           fill=False, 
                                           color='red', 
                                           linestyle='--', 
                                           alpha=0.5)
    plt.gca().add_patch(ellipse)

    # Make axes equal and set tight limits based on grip circle
    plt.axis('equal')
    limit = grip_limit * 1.2  # Add 20% margin
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    
    # Print maximum g-loads
    print(f"\nMaximum G-Loads:")
    print(f"Lateral: {np.max(np.abs(accel_x)):.2f}g")
    print(f"Longitudinal: {np.max(np.abs(accel_y)):.2f}g")
    print(f"Combined: {max_g:.2f}g")

    plt.tight_layout()
    plt.show()

def powerGraph(finalData):
    # Calculate power (P = V * I)
    power = finalData["IVTvoltage"] * finalData["IVTcurrent"] * -1 #T36 IVT voltage is negative

    # Create time array (200 Hz means 1/200 seconds per sample)
    time = np.arange(len(power)) / freqData  # Convert to seconds

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, power)
    plt.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power (Watts)')
    plt.title('Vehicle Power Consumption Over Time')

    # Display the plot
    plt.show()

    return power, time

def energyGraph(finalData, power, time):
    # Calculate energy in kWh by integrating power
    # Power is in Watts, divide by 1000 for kW, divide by 3600 for hours
    dt = 1/freqData  # Time step in seconds
    energy_kwh = np.cumsum(power * dt) / (3600 * 1000)  # Cumulative integration of power to get energy

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, energy_kwh)
    plt.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy (kWh)')
    plt.title('Vehicle Energy Consumption Over Time')

    # Display the plot
    plt.show()

    return energy_kwh

def efficiencyGraph(finalData, power, time):
    # Calculate motor power for each wheel (P = T * ω)
    # Convert RPM to rad/s (multiply by 2π/60)
    rpm_to_rads = 2 * np.pi / 60

    # Calculate power for each wheel
    FRpower = abs(finalData["torqueFR"] * finalData["FRrpm"] * rpm_to_rads)
    FLpower = abs(finalData["torqueFL"] * finalData["FLrpm"] * rpm_to_rads)
    RRpower = abs(finalData["torqueRR"] * finalData["RRrpm"] * rpm_to_rads)
    RLpower = abs(finalData["torqueRL"] * finalData["RLrpm"] * rpm_to_rads)

    # Total motor power
    total_motor_power = FRpower + FLpower + RRpower + RLpower

    # Calculate efficiency (motor power / battery power)
    # Avoid division by zero
    efficiency = np.where(power != 0, total_motor_power / power * 100, 0)

    # Plot efficiency over time
    plt.figure(figsize=(10, 6))
    plt.plot(time, efficiency)
    plt.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Efficiency (%)')
    plt.title('Vehicle Drivetrain Efficiency Over Time')
    plt.ylim(0, 100)  # Limit y-axis to 0-100%

    # Display the plot
    plt.show()

    # Print average efficiency
    print(f"\nAverage Efficiency: {np.median(efficiency):.2f}%")

    return efficiency

def vehiclePathAnimation(finalData):
    # Create animation of vehicle path
    fig = plt.figure(figsize=(10, 10))

    # Downsample data for smoother animation (every 10th point)
    sample_rate = 10
    lat_data = finalData["latitude"][::sample_rate]
    lon_data = finalData["longitude"][::sample_rate]
    current_data = finalData["IVTcurrent"][::sample_rate]

    # Set initial plot limits based on data range
    lon_min, lon_max = np.min(lon_data), np.max(lon_data)
    lat_min, lat_max = np.min(lat_data), np.max(lat_data)
    plt.xlim(lon_min - 0.001, lon_max + 0.001)
    plt.ylim(lat_min - 0.001, lat_max + 0.001)

    # Initialize empty scatter collections for positive and negative current segments
    path_positive = plt.scatter([], [], c='b', alpha=0.8, label='Discharging', s=20)
    path_negative = plt.scatter([], [], c='r', alpha=0.8, label='Regenerating', s=20)
    point = plt.scatter([], [], c='k', s=100, label='Vehicle Position')

    plt.grid(True)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Vehicle Path Animation')

    def init():
        path_positive.set_offsets(np.c_[[], []])
        path_negative.set_offsets(np.c_[[], []])
        point.set_offsets(np.c_[[], []])
        return point, path_positive, path_negative

    def update(frame):
        frame = int(frame)
        if frame < len(lon_data):
            # Update vehicle position
            point.set_offsets(np.c_[[lon_data[frame]], [lat_data[frame]]])
            
            # Split path up to current frame based on current
            pos_mask = current_data[:frame+1] >= 0
            neg_mask = current_data[:frame+1] < 0
            
            # Update positive current points
            path_positive.set_offsets(np.c_[lon_data[:frame+1][pos_mask], 
                                        lat_data[:frame+1][pos_mask]])
            
            # Update negative current points
            path_negative.set_offsets(np.c_[lon_data[:frame+1][neg_mask], 
                                        lat_data[:frame+1][neg_mask]])
        
        return point, path_positive, path_negative

    # Create animation with slower update rate
    anim = FuncAnimation(fig, update,
                        frames=len(lon_data),
                        init_func=init,
                        blit=True,
                        interval=100)

    plt.legend()
    plt.tight_layout()

    # Display the animation
    plt.show()

def vibrationAnalysis(finalData):
# Get acceleration data and convert to g's
    accel_x = finalData["accelX"]/9.81  # Convert to g's
    accel_y = finalData["accelY"]/9.81
    accel_z = finalData["accelZ"]/9.81

    # Calculate magnitude of acceleration vector in g's using Milliken's method
    accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

    # Apply window function to reduce spectral leakage
    window = np.hanning(len(accel_x))
    accel_x *= window
    accel_y *= window
    accel_z *= window
    accel_magnitude *= window

    # Perform FFT on each axis
    n = len(accel_x)
    freq = np.fft.fftfreq(n, d=1/freqData)
    pos_freq_mask = freq >= 0  # Get only positive frequencies

    # Calculate FFT
    fft_x = np.fft.fft(accel_x)
    fft_y = np.fft.fft(accel_y)
    fft_z = np.fft.fft(accel_z)
    fft_mag = np.fft.fft(accel_magnitude)

    # Calculate Power Spectral Density (g²/Hz)
    scale_factor = 2.0 / (freqData * np.sum(window**2))
    psd_x = np.abs(fft_x)**2 * scale_factor
    psd_y = np.abs(fft_y)**2 * scale_factor
    psd_z = np.abs(fft_z)**2 * scale_factor
    psd_mag = np.abs(fft_mag)**2 * scale_factor

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Vehicle Vibration Analysis', fontsize=16)
    
    # Plot PSDs with proper units
    def plot_psd(ax, freq, psd, title):
        ax.semilogy(freq[pos_freq_mask], psd[pos_freq_mask])
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (g²/Hz)')
        ax.grid(True, which="both")
        ax.set_xlim(0, freqData/2)  # Nyquist frequency

    plot_psd(ax1, freq, psd_x, 'Lateral Acceleration PSD')
    plot_psd(ax2, freq, psd_y, 'Longitudinal Acceleration PSD')
    plot_psd(ax3, freq, psd_z, 'Vertical Acceleration PSD')
    plot_psd(ax4, freq, psd_mag, 'Combined G-Load PSD')

    plt.tight_layout()
    plt.show()

    # Calculate RMS g-levels in different frequency bands
    def get_band_rms(psd, freq, f_min, f_max):
        mask = (freq >= f_min) & (freq <= f_max)
        return np.sqrt(np.sum(psd[mask]) * (freq[1]-freq[0]))

    # Calculate and print RMS g-levels in different frequency bands
    freq_bands = [(0,5), (5,20), (20,50), (50,100)]
    print("\nRMS G-Levels in Frequency Bands:")
    for f_min, f_max in freq_bands:
        print(f"\n{f_min}-{f_max} Hz band:")
        print(f"Lateral: {get_band_rms(psd_x[pos_freq_mask], freq[pos_freq_mask], f_min, f_max):.3f}g RMS")
        print(f"Longitudinal: {get_band_rms(psd_y[pos_freq_mask], freq[pos_freq_mask], f_min, f_max):.3f}g RMS")
        print(f"Vertical: {get_band_rms(psd_z[pos_freq_mask], freq[pos_freq_mask], f_min, f_max):.3f}g RMS")
        print(f"Combined: {get_band_rms(psd_mag[pos_freq_mask], freq[pos_freq_mask], f_min, f_max):.3f}g RMS")

    # Make sure to return all values before the function ends
    return freq[pos_freq_mask], psd_x[pos_freq_mask], psd_y[pos_freq_mask], psd_z[pos_freq_mask], psd_mag[pos_freq_mask]

def pacejka_model(x, B, C, D, E):
    """
    Pacejka's Magic Formula tire model
    B: stiffness factor
    C: shape factor
    D: peak value
    E: curvature factor
    """
    return D * np.sin(C * np.arctan(B * x - E * (B * x - np.arctan(B * x))))

def steeringAnalysis(finalData):
    # Convert acceleration to g's and get speed data
    lateral_g_full = finalData["accelX"]/9.81
    steering_full = finalData["steeringAngle"]
    speed_full = np.abs(finalData["velocityY"])
    
    # Create masks for proper cornering behavior
    proper_turning_mask = ((steering_full > 0) & (lateral_g_full < 0)) | ((steering_full < 0) & (lateral_g_full > 0))
    
    # Apply proper turning mask
    lateral_g_full = -1*lateral_g_full[proper_turning_mask]
    steering_full = steering_full[proper_turning_mask]
    speed_full = speed_full[proper_turning_mask]
    
    print(f"\nTotal data points after filtering for proper cornering: {len(steering_full)}")
    
    # Rest of your existing speed range analysis
    best_r2 = 0
    best_speed_range = None
    best_params = None
    best_data = None
    
    # Speed range parameters
    min_speed = 2.0
    max_speed = 33
    speed_step = 0.1      # Increased step size
    range_step = 0.1      # Increased range step
    min_range_width = 1 # Minimum speed range width
    
    # Sweep through speed ranges
    for low_speed in np.arange(min_speed, max_speed-min_range_width, speed_step):
        for range_width in np.arange(min_range_width, max_speed-low_speed, range_step):
            high_speed = low_speed + range_width
            
            # Create mask for current speed range only
            speed_mask = (speed_full >= low_speed) & (speed_full <= high_speed)
            
            # Require sufficient data points
            if np.sum(speed_mask) < 200:
                continue
                
            # Filter data based on speed only
            lateral_g = lateral_g_full[speed_mask]
            steering = steering_full[speed_mask]
            filtered_speed = speed_full[speed_mask]
            
            try:
                # Adjusted Pacejka parameters
                p0 = [15.0, 1.9, 1.6, 0.7]  # Initial guess [B, C, D, E]
                bounds = ([1.0, 1.0, 1.0, 0.2],     # Lower bounds
                         [25.0, 2.5, 2.0, 0.9])     # Upper bounds
                
                popt, _ = curve_fit(pacejka_model, steering, lateral_g,
                                  p0=p0, bounds=bounds, 
                                  maxfev=10000,
                                  method='trf')
                
                B, C, D, E = popt
                
                # Calculate R-squared
                lateral_g_pred = pacejka_model(steering, B, C, D, E)
                r2 = r2_score(lateral_g, lateral_g_pred)
                
                # Update if better fit found
                if r2 > best_r2:
                    best_r2 = r2
                    best_speed_range = (low_speed, high_speed)
                    best_params = (B, C, D, E)
                    best_data = (steering, lateral_g, filtered_speed)
                    
                    print(f"\nNew best fit found:")
                    print(f"Speed range: {low_speed:.1f} - {high_speed:.1f} m/s")
                    print(f"R-squared: {r2:.3f}")
                    
            except RuntimeError:
                continue

    # Always create visualization, even if fit isn't optimal
    # Use the data from best fit if found, otherwise use filtered data
    if best_r2 > 0:
        steering, lateral_g, filtered_speed = best_data
        B, C, D, E = best_params
    else:
        # Use all valid cornering data if no good fit was found
        steering = steering_full
        lateral_g = lateral_g_full
        filtered_speed = speed_full
        
    # Calculate correlation coefficient
    correlation = np.corrcoef(steering, lateral_g)[0,1]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(steering, lateral_g, 
                        alpha=0.5,
                        c=np.abs(lateral_g),
                        cmap='hot',
                        s=10)
    
    # Add Pacejka fit line if parameters were found
    if best_r2 > 0:
        steering_line = np.linspace(np.min(steering), np.max(steering), 100)
        lateral_g_fit = pacejka_model(steering_line, B, C, D, E)
        ax.plot(steering_line, lateral_g_fit, 'b-', 
                linewidth=2, 
                label=f'Pacejka Model (R²={best_r2:.3f})')
    
    ax.grid(True)
    ax.set_xlabel('Steering Angle (degrees)')
    ax.set_ylabel('Lateral G-force (g)')
    ax.set_title('Steering Angle vs Lateral G-force\n' + 
                f'Correlation: {correlation:.2f}')
    
    if best_r2 > 0:
        ax.legend()
    
    # Add center lines
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='|G-force|')
    
    # Print statistics
    print("\nSteering Analysis:")
    print(f"Total data points: {len(steering)}")
    print(f"Correlation coefficient: {correlation:.3f}")
    print(f"Maximum left steering angle: {np.min(steering):.1f}°")
    print(f"Maximum right steering angle: {np.max(steering):.1f}°")
    print(f"Maximum lateral g-force left: {np.min(lateral_g):.2f}g")
    print(f"Maximum lateral g-force right: {np.max(lateral_g):.2f}g")
    print(f"Average speed: {np.mean(filtered_speed):.1f} m/s")
    print(f"Speed range: {np.min(filtered_speed):.1f} to {np.max(filtered_speed):.1f} m/s")
    
    if best_r2 > 0:
        print("\nPacejka Model Parameters:")
        print(f"B (Stiffness factor): {B:.3f}")
        print(f"C (Shape factor): {C:.3f}")
        print(f"D (Peak value): {D:.3f}")
        print(f"E (Curvature factor): {E:.3f}")
        print(f"R-squared value: {best_r2:.3f}")
    
    plt.tight_layout()
    plt.show()
    
    return steering, lateral_g, best_params if best_r2 > 0 else None

if __name__ == "__main__":
    df, fileVariables = main()
    # Process data
    finalData = assignVaraibles(fileVariables, df)
    
    # Generate all analysis graphs
    gLoadGraph(finalData)
    power, time = powerGraph(finalData)
    energy_kwh = energyGraph(finalData, power, time)
    efficiency = efficiencyGraph(finalData, power, time)
    vehiclePathAnimation(finalData)
    freq, psd_x, psd_y, psd_z, psd_mag = vibrationAnalysis(finalData)
    steering, lateral_g, p = steeringAnalysis(finalData)