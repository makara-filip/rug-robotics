import numpy as np
import matplotlib.pyplot as plt

motor_inertia = 0.02
motor_viscous_coefficient = 0.1
torque_constant = 0.1
motor_resistance = 0.3
motor_inductance = 0.1
electric_constant = 0.1

external_load_torque = 1
target_position = 2

# Proportional-Integral controller configuration
K_p = 0.8 # proportional gain
K_i = 0.5 # integral gain

# Initial conditions
position = 0.0
velocity = 0.0
current = 0.0
error_integral = 0.0

time_step = 0.01
time_max = 10
num_steps = int(time_max / time_step)

time = np.linspace(0, time_max, num_steps)
position_t = np.zeros(num_steps)
velocity_t = np.zeros(num_steps)
voltage_t = np.zeros(num_steps)
current_t = np.zeros(num_steps)

# simulation loop
for step in range(num_steps):
    err = target_position - position
    error_integral += err * time_step

    # PI controller
    voltage = K_p * err + K_i * error_integral

    # calculate the differentials from motor dynamics
    d_current_dt = (voltage - motor_resistance * current - electric_constant * velocity) / motor_inductance
    d_velocity_dt = (torque_constant * current - external_load_torque - motor_viscous_coefficient * velocity) / motor_inertia
    d_position_dt = velocity

    current += d_current_dt * time_step
    velocity += d_velocity_dt * time_step
    position += d_position_dt * time_step

    position_t[step] = position
    velocity_t[step] = velocity
    current_t[step] = current
    voltage_t[step] = voltage

max_requred_voltage = np.max(np.abs(voltage_t))
print("Max required voltage (V):", max_requred_voltage)

plt.figure(figsize=(8, 6))

plt.subplot(4, 1, 1)
plt.title("Motor position")
plt.axhline(target_position, color="green", linestyle="--", label="Target Position")
plt.plot(time, position_t, label="Position (rad)", color="orange")
# plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.title("Motor angular velocity")
plt.plot(time, velocity_t, label="Velocity (rad/s)")
# plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.title("Motor armature current")
plt.plot(time, current_t, label="Current (A)", color="red")
# plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.title("Motor voltage")
plt.plot(time, voltage_t, label="Voltage (V)", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)

plt.tight_layout()
plt.show()
