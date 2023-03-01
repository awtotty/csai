import gymnasium as gym
import numpy as np


MAX_STEPS = 1e8
RENDER_FREQ = 100000
ENV_NAME = 'CliffWalking-v0'


def main(): 
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="./video", episode_trigger=lambda x:x%RENDER_FREQ==0, video_length=50,)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    gamma = 1
    alpha = 0.8
    eps = 1
    eps_decay = 1-1e5
    eps_min = 0.05

    # learning
    steps = 0
    episode = 0
    while steps < MAX_STEPS: 
        s, _ = env.reset()
        term = False
        trunc = False
        reward = 0
        while not term and not trunc: 
            a = env.action_space.sample() if np.random.random() < eps else np.argmax(Q[s])
            s_prime, r, term, trunc, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_prime]) - Q[s, a])

            s = s_prime
            reward += r
            steps += 1

            eps *= eps_decay
            if eps < eps_min: 
                eps = eps_min

            if steps >= MAX_STEPS: 
                break

        episode += 1

        if episode % 1000 == 0: 
            print(f"Episode {episode} reward: {reward}")

    
if __name__ == "__main__": 
    main()