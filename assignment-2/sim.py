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
    M = get_mass_inertia_matrix(configuration)
    C = get_coriolis_forces(configuration, config_derivative_first)
    D = get_dissipative_forces()
    G = get_gravity_vector(configuration)
    torques = M @ config_derivative_second + C @ config_derivative_first + D @ config_derivative_first + G
    return torques

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
