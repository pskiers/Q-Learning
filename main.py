import gym
from Qlearning import q_learning
import numpy as np
from matplotlib import pyplot as plt


def main():
    env = gym.make('FrozenLake-v1')
    episodes = 80000
    max_steps = 100
    learning_rate = 0.15
    discount_rate = 0.99
    exploration_rate = 1
    exploration_decay = 0.0001
    max_exploration = 1
    min_exploration = 0.001

    q, history = q_learning(
        env=env,
        learning_rate=learning_rate,
        episodes=episodes,
        discount_rate=discount_rate,
        exploration_rate=exploration_rate,
        max_steps=max_steps,
        exploration_decay=exploration_decay,
        max_exploration=max_exploration,
        min_exploration=min_exploration
    )
    per = 1000
    avg_rewards = np.split(np.array(history), episodes/per)
    avg_rewards = [sum(i/per) for i in avg_rewards]
    for i, rew in enumerate(avg_rewards):
        print('Iteration: ', per*(i+1), '\tAverage reward: ', rew)
    x = [i * per for i in range(len(avg_rewards))]
    plt.plot(x, avg_rewards)
    plt.show()



if __name__ =="__main__":
    main()