import numpy as np
import matplotlib.pyplot as plt

motor_inertia = 0.02
motor_viscous_coefficient = 0.1
motor_inductance = 0.1
motor_resistance = 0.3
electric_constant = 0.1
torque_constant = 0.1

current_limit = 60
target_position = 1

position = 0
velocity = 0
current = 0
voltage = 0

time_step = dt = 0.001
time_max = 3
steps = int(time_max / time_step)

time = np.linspace(0, time_max, steps)
position_t = np.zeros(steps)
velocity_t = np.zeros(steps)
current_t = np.zeros(steps)
voltage_t = np.zeros(steps)

# PI controller gains
# current
Kp_i, Ki_i = motor_inductance / time_step, motor_resistance / time_step
# velocity
Kp_w, Ki_w = motor_inertia / time_step / 4, motor_viscous_coefficient / time_step / 4
# position
Kp_theta, Ki_theta = 5.0, 2.0

# Control loop simulation
for step in range(steps):
    # position control
    position_error = target_position - position
    velocity_ref = Kp_theta * position_error + Ki_theta * position_error * dt
    
    # velocity control
    velocity_error = velocity_ref - velocity
    current_ref = Kp_w * velocity_error + Ki_w * velocity_error * dt
    
    # feedforward current term for inertia
    current_ff = motor_inertia * velocity_ref / torque_constant
    current_ref += current_ff
    
    # current control
    current_error = current_ref - current
    voltage = Kp_i * current_error + Ki_i * current_error * dt
    
    if abs(current) > current_limit:
        current = np.sign(current) * current_limit
    
    d_position = velocity * dt
    d_velocity = (torque_constant * current - motor_viscous_coefficient * velocity) / motor_inertia * dt
    d_current = (voltage - motor_resistance * current - electric_constant * velocity) / motor_inductance * dt

    position += d_position
    velocity += d_velocity
    current += d_current
    
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
plt.axhline(current_limit, linestyle="--", label="Current limit", color="red")
plt.axhline(-current_limit, linestyle="--", color="red")
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
