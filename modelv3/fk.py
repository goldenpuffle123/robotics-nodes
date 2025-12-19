import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
DH_BAXTER = np.array([ # DH parameters given in paper.
    # d,        a,      alpha
    [0.27035,   0.069,  -np.pi/2],  # Link 1 (S0-S1)
    [0.0,       0.0,     np.pi/2],  # Link 2 (S1-E0)
    [0.36435,   0.069,  -np.pi/2],  # Link 3 (E0-E1)
    [0.0,       0.0,     np.pi/2],  # Link 4 (E1-W0)
    [0.37429,   0.01,   -np.pi/2],  # Link 5 (W0-W1)
    [0.0,       0.0,     np.pi/2],  # Link 6 (W1-W2)
    [0.229525,  0.0,     0.0]       # Link 7 (W2-EE)
])

# We also need the theta offsets (second column)
THETA_OFFSETS_BAXTER = np.array([
    0.0,
    np.pi/2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
])

def get_tmatrix(d, a, alpha, theta):
    """Compute the individual transformation matrix using DH parameters."""
    return np.array([ # Form of the transformation matrix given in the paper.
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])
def calculate_fk(joint_angles):
    """Calculate the forward kinematics for the Baxter robot given joint angles."""
    T_final = np.eye(4)
    
    # Iterate through each joint
    for i in range(len(joint_angles)):
        # Get the parameters for this link
        q = joint_angles[i] + THETA_OFFSETS_BAXTER[i] # Add the offset
        d, a, alpha = DH_BAXTER[i]
        
        # Get the transformation matrix for this link
        T_link = get_tmatrix(d, a, alpha, q)
        # Cumultatively multiply to get the final transformation
        T_final = T_final @ T_link
        
    return T_final

if __name__ == "__main__":
    pass