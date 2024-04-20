import csv
import numpy as np
import lab3
import matplotlib.pyplot as plt



if __name__ == '__main__':
    env = lab3.gym.make('MountainCar-v0', render_mode=None)
    position = np.linspace(-1.2, 0.6, 20)
    velocity = np.linspace(-0.07, 0.07, 20)

    q_table = np.load('q_table.npy')
    print(q_table)
    ans = 0
    n = 10

    for k in range(n):
        state = env.reset()
        discrete_state = lab3.discrete(state[0], position, velocity)
        done = False
        total_reward = 0

        for i in range(1000):
            action = np.argmax(q_table[discrete_state])
            step = env.step(action)
            new_state = step[0]
            new_discrete_state = lab3.discrete(new_state, position, velocity)
            reward = step[1]
            done = step[2]

            discrete_state = new_discrete_state

            if done :
                ans += 1
                break

            total_reward += reward
        print(f"Episode {k+1}: Total Reward: {total_reward}")

    print("Accurancy", (ans / n) * 100)
    q_table_mean = np.mean(q_table, axis=2)  

    plt.imshow(q_table_mean, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Heatmap of Q-values")
    plt.show()

    
    env.close()

