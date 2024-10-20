import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Function to get the position of each joint and the end-effector
def get_joint_positions(configuration):
    joint_angle_1, joint_angle_2 = configuration
    
    # Position of the first joint
    x1 = link_length_1 * math.cos(joint_angle_1)
    y1 = link_length_1 * math.sin(joint_angle_1)
    
    # Position of the end-effector (second joint)
    x2 = x1 + link_length_2 * math.cos(joint_angle_1 + joint_angle_2)
    y2 = y1 + link_length_2 * math.sin(joint_angle_1 + joint_angle_2)
    
    return (0, 0), (x1, y1), (x2, y2)

# Function to animate the robotic arm over time
def animate_robot_arm(t_vals, theta1_vals, theta2_vals):
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title('Planar Robotic Arm Simulation')
    
    # Line representing the robotic arm's links
    line, = ax.plot([], [], 'o-', lw=4, markersize=8)
    
    def update(frame):
        # Get the current angles
        theta1 = theta1_vals[frame]
        theta2 = theta2_vals[frame]
        
        # Compute the joint positions
        base, joint1, end_effector = get_joint_positions((theta1, theta2))
        
        # Update the line data
        line.set_data([base[0], joint1[0], end_effector[0]], [base[1], joint1[1], end_effector[1]])
        
        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(t_vals), interval=20, blit=True)
    plt.show()

# Main function to run the simulation and visualize the robotic arm
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
    theta1_vals = solution.y[0]
    theta2_vals = solution.y[1]

    # Visualize the robotic arm over time
    animate_robot_arm(solution.t, theta1_vals, theta2_vals)

# Run the main function
if __name__ == "__main__":
    main()
