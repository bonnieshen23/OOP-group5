import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

#===============================================
#2025/12/06 張翊萱
#In run(), I changed the inital value of q by a small value(0.001) instead of zero.
#This let us to update the q value no matter the robot is reaching the goal or not.
#I use Expected SARSA algorithm to improve the performace.

#Q-learning:
#Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', A)) - Q(s, a))
# s: current state, s': next state, a: action, A: set of all possible action in next state, α: learning rate, γ: discount_factor

#SARSA:
#Q(s, a) = Q(s, a) + α * (r + γ * Q(s', a') - Q(s, a))
# s: current state, s': next state, a: action, a': next action, α: learning rate, γ: discount_factor

#Expected SARSA:
#Q(s, a) = Q(s, a) + α * (r + γ * ∑ π(a|s') * Q(s', a) - Q(s, a))       
# s: current state, s': next state, π: posibility, a: action, α: learning rate, γ: discount_factor
# -- ∑ π(a|s') * Q(s', a) is the expection of the action --

#I also mutiply a rate to q value to stop the robot from falling into ice holes.
#===============================================

def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False, show_result = True):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.ones((env.observation_space.n, env.action_space.n)) * 0.001 # init a 64 x 4 array with value of 0.001
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.95 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    falling_discount_rate = 0.5        # mutiply this to prevent the robot fall into ice holes
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall into holes or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * (q[new_state, np.argmax(q[new_state,:])] * (1 - epsilon) + np.sum(q[new_state, :]) * epsilon * 0.25)  #expection
                    - q[state,action])
                
                # stop the robot from falling into ice holes
                if terminated and reward != 1:    
                    q[state, action] *= falling_discount_rate    
            
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

        #output the cause of termination
        if show_result:
            if reward == 1:
                    print("Terminated: get reward")
                    rewards_per_episode[i] = 1

            elif terminated:
                print("Terminated: fall into ice hole")
            
            else:
                print("Truncated: over 200 steps")

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    
    if is_training == False:
        rate = print_success_rate(rewards_per_episode)
        return rate

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    #training
    run(15000, is_training=True, render=False)

    #testing
    run(10, is_training=False, render=False, show_result = False)