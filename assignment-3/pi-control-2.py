import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# define motor constants
motor_inertia = 0.02
motor_viscous_coefficient = 0.1
torque_constant = 0.1
motor_resistance = 0.3
motor_inductor = 0.1
electric_constant = 0.1

desired_velocity = 10
maximum_velocity = desired_velocity * 1.5

def get_velocity_transform_function():
    numerator = [torque_constant]
    denominator = [
        # s^2 term:
        motor_inertia * motor_inductor,
        # s term:
        motor_resistance * motor_inertia + motor_viscous_coefficient * motor_inductor,
        # constant term:
        motor_viscous_coefficient * motor_resistance + electric_constant * torque_constant
    ]
    return ctrl.TransferFunction(numerator, denominator)

# motor angular velocity is considered as the system output
motor_velocity_tf = get_velocity_transform_function()

# PI controller design
Kp = 6 # proportional gain
Ki = 4 # integral gain

# control(s) = Kp + Ki/s = (Kp*s + Ki) / s
controller_tf = ctrl.TransferFunction([Kp, Ki], [1, 0])

closed_loop_velocity_tf = ctrl.feedback(motor_velocity_tf * controller_tf)

# time simulation
time_start = 0
time_end = 2
time_step = 0.001
time_vector = np.arange(time_start, time_end, time_step)

t_out, velocity_response = ctrl.step_response(desired_velocity * closed_loop_velocity_tf, time_vector)

# integrate the velocity over time to determine position
position_response = np.cumsum(velocity_response) * time_step

# calculate the required voltage for the response
# U(s) = control(s) * error(s)
# error(s) = desired_velocity - actual_velocity
# voltage(t) = Kp * error(t) + Ki * integral_0^t error(dt) dt
error_signal = desired_velocity - velocity_response
error_integral = np.cumsum(error_signal) * time_step

voltage_t = Kp * error_signal + Ki * error_integral
max_voltage_used = np.max(np.abs(voltage_t))

print("Proportional gain (Kp):", Kp)
print("Integral gain (Ki):", Ki)
print("Maximum required voltage:", max_voltage_used)

plt.figure(figsize=(8, 6))


plt.subplot(3, 1, 1)
plt.title("Control voltage input to motor")
plt.plot(t_out, voltage_t, label="Control voltage (V)", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.title("Velocity response of the DC motor with PI control")
plt.plot(t_out, velocity_response, label="Motor velocity (rad/s)")
plt.axhline(desired_velocity, color="green", linestyle="--", label="Desired velocity")
plt.axhline(maximum_velocity, color="red", linestyle="--", label="Maximum velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.title("Angular position of the DC Motor")
plt.plot(t_out, position_response, label="Motor position (rad)", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
