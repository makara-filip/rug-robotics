import math
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
# Link lengths
L1, L2, L3 = 2.0, 2.0, 1.0


# Gain for the Jacobian transpose method
alpha = 0.02
iterations = 500
t = np.arange(0, iterations, 1) # Time array

# Desired end-effector position and orientation
x_destination = 2.0 * math.sqrt(2)
y_destination = 2.0 * math.sqrt(2) + 1
rotation_destination = math.pi / 2 # 90 degrees

# Initialize joint angles
q = np.zeros((3, len(t))) # Store joint angles over time
q[:, 0] = [0.0, 0.0, 0.0] # Initial joint angles

def forward_kinematics(configuration):
    theta1, theta2, theta3 = configuration
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2) + L3 * np.sin(theta1 + theta2 + theta3)
    phi = theta1 + theta2 + theta3
    return np.array([x, y, phi])

# Jacobian function
def jacobian(configuration):
    theta1, theta2, theta3 = configuration
    J = np.array([
        [-L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2) - L3 * np.sin(theta1 + theta2 + theta3), 
         -L2 * np.sin(theta1 + theta2) - L3 * np.sin(theta1 + theta2 + theta3), 
         -L3 * np.sin(theta1 + theta2 + theta3)],
        [L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3), 
         L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3), 
         L3 * np.cos(theta1 + theta2 + theta3)],
        [1, 1, 1]
    ])
    return J

# Iterative algorithm to simulate joint movement over time
for iter in range(iterations-1):
    current_configuration = q[:, iter]
    x, y, rotation = forward_kinematics(current_configuration)

    error_vector = np.array([
        x_destination - x,
        y_destination - y,
        rotation_destination - rotation
    ])
    
    # Update joint angles using Jacobian transpose method
    step = alpha * jacobian(current_configuration).T @ error_vector
    q[:, iter + 1] = current_configuration + step

final_configuration = q[:, iterations-1]
print("Inverse kinematics solution:")
x, y, rotation = forward_kinematics(final_configuration)
print("x =", x)
print("y =", y)
print("rot =", rotation)

# Plot the joint variables over time
plt.plot(t, q[0, :], label=r'$\theta_1(t)$')
plt.plot(t, q[1, :], label=r'$\theta_2(t)$')
plt.plot(t, q[2, :], label=r'$\theta_3(t)$')
plt.xlabel('Time [s]')
plt.ylabel('Joint Angles [rad]')
plt.legend()
plt.title('Joint Angle Trajectories')
plt.grid()
plt.show()
