import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

link_mass = 5.0
link_arm_length = 0.5
link_inertia = 25
link_friction = 25
motor_inertia = 2
motor_friction = 5
reduction_ratio = 10
gravitational_acceleration = 9.81

total_friction = link_friction + motor_friction * (reduction_ratio ** 2)
total_inertia = link_inertia + motor_inertia * (reduction_ratio ** 2)

target_position = np.radians(45)

# PD controller gains (chosen through trial)
Kp = 600.0
Kd = 200.0

# Define the dynamics of the pendulum with PD control
def pendulum_dynamics(t, y):
    position, velocity = y
    error = target_position - position
    # torque = Kp*err - Kd*velocity + g(position)
    gravity = link_mass * gravitational_acceleration * link_arm_length * np.sin(position)
    torque_control = Kp * error - Kd * velocity + gravity
    
    # Equation of motion for the pendulum with control torque
    acceleration = (torque_control - total_friction * velocity - gravity) / total_inertia
    return [velocity, acceleration]

time_min = 0
time_max = 8
time_step = 0.001
time_span = (time_min, time_max)
time_points_to_evaluate = np.arange(time_min, time_max, time_step)

initial_position = 0.0
initial_velocity = 0.0
initial_state = [initial_position, initial_velocity]

solution = solve_ivp(pendulum_dynamics, time_span, initial_state, t_eval=time_points_to_evaluate)
position_t = solution.y[0]
velocity_t = solution.y[1]
time = solution.t

error_t = target_position - position_t
gravity_t = link_mass * gravitational_acceleration * link_arm_length * np.sin(position_t)
torque_control_t = Kp * error_t - Kd * velocity_t + gravity_t

final_torque = torque_control_t[-1]
print("Final torque:", final_torque)

# Plotting results
plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.title('Link angle over time')
plt.axhline(np.degrees(target_position), color="green", linestyle='--', label="Desired position")
plt.plot(time, np.degrees(position_t), label='Joint position', color="orange")
plt.xlabel('Time (s)')
plt.ylabel('Joint angle (deg)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Control torque over time')
plt.plot(time, torque_control_t, label='Control torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
