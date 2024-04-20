import gymnasium as gym
import pygame
import numpy as np
import random


def discrete(state, position, velocity):
    pos, vel = state
    position_index = np.digitize(pos, position)  
    velocity_index = np.digitize(vel, velocity)
    return (position_index, velocity_index)


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode=None)
    position = np.linspace(-1.2, 0.6, 20)
    velocity = np.linspace(-0.07, 0.07, 20)

    epsilon = 1
    learning_rate = 0.5
    discount_factor = 0.95

    table_size = (20, 20, 3)
    q_table = np.zeros(table_size)
    print(q_table)
    all_rewards = []
    max_reward = 0.0
    done_cnt = 0

    for k in range(1, 50000):
        done = False
        state = env.reset()
        discrete_state = discrete(state[0], position, velocity)
        total = 0
        old_reward = 0
        
        for i in range(1000):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[discrete_state])
            
            step = env.step(action)
            new_state = step[0]
            new_discrete_state = discrete(new_state, position, velocity)
            old_reward += step[1]
            reward = step[1]
            done = step[2]

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]

                new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
                q_table[discrete_state + (action,)] = new_q
                total += reward
            
            discrete_state = new_discrete_state

            if done:
                done_cnt += 1
                break
        
        if (k % 100) == 0 and learning_rate > 0.01:
            learning_rate -= 0.001
            
        if (k % 100) == 0 and epsilon > 0.1:  
            epsilon -= 0.0025
        
        if (k % 100) == 0:
            print("Episode:", k, " reward:", total, "old reward:", old_reward, "Episode done:", done_cnt, "eps:", epsilon, "lr:", learning_rate)
            np.save('q_table.npy', q_table)

    np.save('q_table.npy', q_table)
    print(q_table)
    env.close()

