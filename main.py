import numpy as np
from environment import grid_world
from visualize import draw_image

WORLD_SIZE = 5      # set the size of grid
# left, up, right, down
ACTIONS = {'LEFT':np.array([0, -1]), 'UP':np.array([-1, 0]), 'RIGHT':np.array([0, 1]), 'DOWN':np.array([1, 0])}
ACTION_PROB = 0.25      # set equiprobable policy of action



def evaluate_state_value_by_matrix_inversion(env, discount=0.9): # set discount rate as 0.9
    WIDTH, HEIGHT = env.size()      # set environment size as 5 by 5

    # Reward matrix R
    R = np.zeros((WIDTH, HEIGHT))       #create zero matrix for reward
    for i in range(WIDTH):
        for j in range(HEIGHT):
            expected_reward = 0         #initialize reward matrix as 0
            for action in ACTIONS:
                (next_i, next_j), reward = env.interaction([i, j], ACTIONS[action])
                expected_reward += ACTION_PROB*reward       # set expectation of reward
            R[i, j] = expected_reward       #save expected reward data in matrix
    R = R.reshape((-1,1))       # convert 1D array to 2D array

    # Transition matrix T
    
    T = np.zeros((WIDTH*HEIGHT, WIDTH*HEIGHT))          #create matrix to save current and next state value
    for i in range(WIDTH):          #iteation for width which is column of T matrix
        for j in range(HEIGHT):         # iteration for height which is row of T matrix
            current_state = i * WIDTH + j       # save current state as 1D data
            for action in ACTIONS:      # calculate for all four actions
                (next_i, next_j), reward = env.interaction([i,j], ACTIONS[action])      #calculate for next state
                next_state = next_i * WIDTH + next_j        #save next state as 1D data

                T[current_state, next_state] += ACTION_PROB         # calculate expectation for action prob

    I = np.eye(WIDTH * HEIGHT)      # determine diagonal matrix for matrix calculation
    V = np.linalg.inv(I - discount * T)@R       # determine V = (I-rT)^-1R which is state value calculation

    new_state_values = V.reshape(WIDTH,HEIGHT)      # determine next state value using V matrix values
    draw_image(1, np.round(new_state_values, decimals=2))   # plot the value functions

    return new_state_values


if __name__ == '__main__':          # check if it is main file
    env = grid_world()      # load environment file from grd_world class
    values = evaluate_state_value_by_matrix_inversion(env = env)        #



