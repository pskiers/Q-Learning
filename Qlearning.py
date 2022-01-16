import numpy as np
import random
import gym


def q_learning(env: gym.Env,
               learning_rate: float,
               episodes: int,
               discount_rate: float,
               exploration_rate: float,
               max_steps=None,
               exploration_decay=0,
               max_exploration=1,
               min_exploration=0):

    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    Q = np.zeros((state_space_size, action_space_size))
    history = []
    print('_____________________Training________________________')

    for episode in range(episodes):
        state = env.reset()
        done = False
        curr_rewards = 0
        step = 1
        while(not done):
            if max_steps is not None:
                if max_steps < step:
                    break
            exp = random.uniform(0, 1)
            if exp > exploration_rate:
                action = np.argmax(Q[state,:])
            else:
                action = env.action_space.sample()
            new_state, reward, done, other = env.step(action)
            Q[state, action] = Q[state, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(Q[new_state, :]))
            state = new_state
            curr_rewards += reward

        exploration_rate = min_exploration + (max_exploration - min_exploration) * np.exp(-exploration_decay * episode)
        history.append(curr_rewards)
        if episode % 2000 == 0:
            print('Episode: ', episode)

    print('____________________Training_Ended____________________')
    return Q, history
