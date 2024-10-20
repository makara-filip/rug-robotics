import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

print("Imported.")

link_length_1, link_length_2 = 2.0, 1.0
link_mass_1, link_mass_2 = 50.0, 25.0
link_inertia_1, link_inertia_2 = 10.0, 5.0
center_of_mass_1 = link_length_1 / 2
center_of_mass_2 = link_length_2 / 2

joint_friction_1, joint_friction_2 = 50.0, 25.0
gravity_constant = 9.81

def radian(angle_in_degrees):
    return angle_in_degrees / 360 * 2 * np.pi

def end_effector_position(configuration):
    joint_angle_1, joint_angle_2 = configuration

    return np.array([
        [
            link_length_1 * math.cos(joint_angle_1) + link_length_2 * math.cos(joint_angle_1 + joint_angle_2),
            link_length_1 * math.sin(joint_angle_1) + link_length_2 * math.sin(joint_angle_1 + joint_angle_2),
            0
        ]
    ])

def get_mass_inertia_matrix(configuration):
    joint_angle_1, joint_angle_2 = configuration

    return np.array([
        [
            link_inertia_1 + link_inertia_2 + link_mass_1*(center_of_mass_1**2) + link_mass_2*(link_length_1**2 + center_of_mass_2**2 + 2*link_length_1*center_of_mass_2*np.cos(joint_angle_2)),
            link_inertia_2 + link_mass_2*(center_of_mass_2**2 + link_length_1*center_of_mass_2*np.cos(joint_angle_2))
        ],
        [
            link_inertia_2 + link_mass_2*(center_of_mass_2**2 + link_length_1*center_of_mass_2*np.cos(joint_angle_2)),
            link_inertia_2 + link_mass_2*(center_of_mass_2**2)
        ]
    ])

def get_coriolis_forces(configuration, configuration_derivative):
    joint_angle_1, joint_angle_2 = configuration
    joint_angle_1_derivative, joint_angle_2_derivative = configuration_derivative

    scalar_multiplier = - link_mass_2 * link_length_1 * center_of_mass_2 * np.sin(joint_angle_2)
    return scalar_multiplier * np.array([
        [joint_angle_2_derivative, joint_angle_1_derivative + joint_angle_2_derivative],
        [-joint_angle_1_derivative, 0]
    ])

def get_dissipative_forces():
    return np.array([
        [joint_friction_1, 0],
        [0, joint_friction_2]
    ])

def get_gravity_vector(configuration):
    joint_angle_1, joint_angle_2 = configuration

    return np.array([
        (link_mass_1*center_of_mass_1 + link_mass_2*link_length_1) * gravity_constant * np.cos(joint_angle_1)
        + link_mass_2 * center_of_mass_2 * gravity_constant * np.cos(joint_angle_1 + joint_angle_2),

        link_mass_2 * center_of_mass_2 * gravity_constant * np.cos(joint_angle_1 + joint_angle_2)
    ])

def joint_torques_forces(configuration, config_derivative_first, config_derivative_second):
    # ignoring external forces

    M = get_mass_inertia_matrix(configuration)
    C = get_coriolis_forces(configuration, config_derivative_first)
    D = get_dissipative_forces()
    G = get_gravity_vector(configuration)

    torques = M @ config_derivative_second + (C + D) @ config_derivative_first + G
    return torques

initial_config = (0.0, 0.0)
initial_config_derivative_first = (0.0, 0.0)
initial_config_derivative_second = (0.0, 0.0)

print(
    joint_torques_forces(
        initial_config,
        initial_config_derivative_first,
        initial_config_derivative_second
    )
)

print(
    "3d end-effector position:",
    end_effector_position((
        radian(60),
        radian(-150),
    ))
)


# config = initial_config
# config_derivative_first = initial_config_derivative_first
# config_derivative_second = initial_config_derivative_second
# effect = 0.01

# for iteration in range(1000):



# Function defining the derivatives for solve_ivp
def dynamics(t, y):
    theta1, theta2, theta1_dot, theta2_dot = y
    configuration = np.array([theta1, theta2])
    config_derivative_first = np.array([theta1_dot, theta2_dot])
    
    # Compute the mass matrix and other components
    M = get_mass_inertia_matrix(configuration)
    C = get_coriolis_forces(configuration, config_derivative_first)
    D = get_dissipative_forces()
    G = get_gravity_vector(configuration)
    
    # Assuming zero external torques (can be modified as needed)
    torques = np.array([0, 0])
    
    # Solve for angular accelerations
    config_derivative_second = np.linalg.solve(M, torques - C @ config_derivative_first - D @ config_derivative_first - G)
    
    return [theta1_dot, theta2_dot, config_derivative_second[0], config_derivative_second[1]]

# Main function to run the simulation and plot results
def main():
    # Initial conditions
    initial_config = [radian(0), radian(0)]  # Initial angles in radians
    initial_velocities = [0.0, 0.0]  # Initial angular velocities
    y0 = initial_config + initial_velocities

    # Time span for simulation
    t_span = (0, 10)  # Simulate for 10 seconds
    t_eval = np.linspace(t_span[0], t_span[1], 500)  # Time points to evaluate

    # Solve the system using solve_ivp
    solution = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, method='RK45')

    # Extract results
    theta1 = solution.y[0]
    theta2 = solution.y[1]
    theta1_dot = solution.y[2]
    theta2_dot = solution.y[3]

    # Plot joint angles over time
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(solution.t, theta1, label='Theta 1 (rad)')
    plt.plot(solution.t, theta2, label='Theta 2 (rad)')
    plt.title('Joint Angles')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()

    # Plot joint velocities over time
    plt.subplot(2, 1, 2)
    plt.plot(solution.t, theta1_dot, label='Theta 1 dot (rad/s)')
    plt.plot(solution.t, theta2_dot, label='Theta 2 dot (rad/s)')
    plt.title('Joint Velocities')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()


def manipulator_dynamics(t, y):
    q_1, q_2, qdot_1, qdot_2 = y

    # gravity torques
    tau_g1 = link_mass_1 * gravity_constant * link_length_1/2 * np.sin(q_1)
    tau_g2 = link_mass_2 * gravity_constant * (link_length_1 * np.sin(q_1) + link_length_2/2 * np.sin(q_1 + q_2))

    # friction torques
    tau_f1 = -joint_friction_1 * qdot_1
    tau_f2 = -joint_friction_2 * qdot_2

    # equations of motion for a planar two-link manipulator without external forces
    # inertia matrix
    # M = np.array([
    #     [link_inertia_1 + link_inertia_2 + link_mass_1*(link_length_1/2)**2 + link_mass_2*(link_length_1**2 + (link_length_2/2)**2 + 2*link_length_1*(link_length_2/2)*np.cos(q_2)), link_inertia_2 + link_mass_2*((link_length_2/2)**2 + link_length_1*(link_length_2/2)*np.cos(q_2))],
    #     [link_inertia_2 + link_mass_2*((link_length_2/2)**2 + link_length_1*(link_length_2/2)*np.cos(q_2)), link_inertia_2 + link_mass_2*(link_length_2/2)**2]
    # ])
    M = get_mass_inertia_matrix((q_1, q_2))
    
    # Coriolis forces
    C = np.array([
        [-link_mass_2 * link_length_1 * (link_length_2/2) * np.sin(q_2) * qdot_2, -link_mass_2 * link_length_1 * (link_length_2/2) * np.sin(q_2) * (qdot_1 + qdot_2)],
        [link_mass_2 * link_length_1 * (link_length_2/2) * np.sin(q_2) * qdot_1, 0]
    ])
    
    c = C @ np.array([qdot_1, qdot_2])
    tau_g = np.array([tau_g1, tau_g2])
    tau_f = np.array([tau_f1, tau_f2])
    tau = tau_g + tau_f - c
    accelerations = np.linalg.solve(M, tau)

    # Derivative of state vector
    dq_1_dt, dq_2_dt = qdot_1, qdot_2
    dqdot_1_dt, dqdot_2_dt = accelerations

    return [dq_1_dt, dq_2_dt, dqdot_1_dt, dqdot_2_dt]

# # Initial conditions (q_1, q_2, qdot_1, qdot_2)
# initial_conditions = [0, 0, 0, 0]

# t_span = (0, 10)  # 10 seconds of simulation
# t_eval = np.linspace(0, 10, 300)  # evaluation points

# solution = solve_ivp(manipulator_dynamics, t_span, initial_conditions, method='RK45', t_eval=t_eval)



# # Plot results
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(solution.t, solution.y[0], label='q_1 (rad)')
# plt.plot(solution.t, solution.y[1], label='q_2 (rad)')
# plt.title('Joint Angles over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Angle (rad)')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(solution.t, solution.y[2], label='qdot_1 (rad/s)')
# plt.plot(solution.t, solution.y[3], label='qdot_2 (rad/s)')
# plt.title('Angular Velocities over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Angular Velocity (rad/s)')
# plt.legend()

# plt.tight_layout()
# plt.show()
