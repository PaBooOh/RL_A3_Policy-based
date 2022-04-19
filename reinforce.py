
import numpy as np
import gym
import sys
import time
import signal
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import GradientTape
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline

# (Hyper)parameters
#--------------------------------------------------------Trainning
LR = 0.0001
EPISODES = 2000
OPTIM = Adam(lr=LR)
#--------------------------------------------------------Early-stopping
CONSECUTIVES = 100
OPTIMAL_AVG_STEPS = 250
#--------------------------------------------------------Reinforecement learning
GAMMA = 0.95
#--------------------------------------------------------Environment
ENVNAME = 'CartPole-v1'
#--------------------------------------------------------Plot
avg_list = []
step_list = []
FIG_FILE = 'Reinforce__Lr={}, GAMMA={}, time={}_.png'.format(LR, GAMMA, datetime.now().strftime("%m%d%Y%H%M%S"))

def handler(signum, frame):
    msg = "Ctrl-C was pressed. Screenshot was saved at the current local folder. \n"
    print()
    print(msg, end="", flush=True)
    plotPerformance(step_list)
    sys.exit(1)

def plotPerformance(step_counter_list):
    
    x = np.arange(0, len(step_counter_list))
    y1 = step_counter_list
    # x_sm = np.array(x)
    # y1_sm = np.array(y1)
    # spline = make_interp_spline(x_sm, y1_sm)
    
    # x_smooth = x
    # y1_smooth = spline(x_smooth)
    if len(y1) <= 300:
        window_len = 5
    else:
        window_len = 51
    y1_smooth = savgol_filter(y1, window_len, 2)

    fig, ax1 = plt.subplots()
    ax1.plot(x, y1_smooth, label='Total steps')
    ax1.legend(loc='upper right')

    ax1.set_xlabel('Training episode')
    ax1.set_ylabel('Total steps')
    plt.savefig(FIG_FILE)

class Net():
    def __init__(self, N_STATES, N_ACTIONS) -> None:
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
    def net_fc(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.N_STATES, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.N_ACTIONS, activation='softmax'))
        return model
    
class ReinforceAgent():
    def __init__(self, N_STATES, N_ACTIONS):
        nn = Net(N_STATES, N_ACTIONS)
        self.model = nn.net_fc()
        self.S_buffer = []
        self.A_buffer = []
        self.R_buffer = []
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
    
    def act(self, S): # based on softmax
        S = np.reshape(S, (1, self.N_STATES)) # convert shape such that tf model is able to recognize.
        probs = self.model.predict(S)[0]
        A = np.random.choice(range(self.N_ACTIONS), 1, p=probs)[0]
        return A
    
    def appendBuffer(self, S, A, R):
        S = np.reshape(S, (1, self.N_STATES))
        self.S_buffer.append(S)
        self.A_buffer.append(A)
        self.R_buffer.append(R)

    def calReturns(self):
        returns_G = []
        episode_R = self.R_buffer
        # (1) Gt is calculated based on 1-step target bootstrapping
        cumulative_R = 0
        for R in reversed(episode_R):
            cumulative_R = R + GAMMA * cumulative_R
            returns_G.append(cumulative_R)
        returns_G.reverse()

        # (2) Gt is calculated based on MC 
        # for i in range(len(episode_R)):
        #     return_G = 0
        #     pow = 0
        #     for R in episode_R[i:]:
        #         return_G += R * GAMMA ** pow
        #         pow += 1
        #     returns_G.append(return_G)
        # returns_G = (returns_G - np.mean(returns_G)) / (np.std(returns_G) + 1e-9) # normilization
        return returns_G # R1, R2, ... , Rt.
    
    def calLoss(self, S, A, R): 
        probs = self.model(S)[0]
        log_prob = tf.math.log(probs[A])
        loss = -log_prob * R
        return loss 

    def training(self):
        returns_G = self.calReturns()
        # losses = []
        for S, A, R in zip(self.S_buffer, self.A_buffer, returns_G):
            with GradientTape() as gt:
                loss = self.calLoss(S, A, R)
                grads = gt.gradient(loss, self.model.trainable_variables)
                OPTIM.apply_gradients(zip(grads, self.model.trainable_variables))
                # losses.append(loss)

        self.S_buffer = []
        self.A_buffer = []
        self.R_buffer = []
        # return np.mean(losses)

def main():
    global avg_list # For plotting
    global step_list
    ENV = gym.make(ENVNAME)  # make game env
    N_STATES = ENV.observation_space.shape[0] # 4
    N_ACTIONS = ENV.action_space.n # 2
    reinforce_agent = ReinforceAgent(N_STATES, N_ACTIONS)
    step_counts_list = [] # for recording the number of the step the agent do in each game
    scores = [] # for recording the cumulative rewards in each game
    win_most_recent = deque(maxlen=CONSECUTIVES) # for 'early-stopping'
    # Playing
    for episode in range(EPISODES):
        S = ENV.reset()
        step_counts = 0
        total_R = 0 # the sum of the rewards in a game
        # Early-stopping
        if np.mean(win_most_recent) >= OPTIMAL_AVG_STEPS:
            print('Reach good model')
            break
        # Start collecting data
        while True:
            # ENV.render()
            A = reinforce_agent.act(S) # take an action based on softmax
            S_next, R, terminal, _ = ENV.step(A) # perform an action and get corresponding info about the next state
            step_counts += 1
            if terminal:
                if step_counts == 500:
                    R = 10
                elif step_counts < 500:
                    R = -R # If fail, suffer punishment
            else:
                R = R
            reinforce_agent.appendBuffer(S, A, R)
            S = S_next
            total_R += R

            # End if there are enough winnings.
            if terminal:
                win_most_recent.append(step_counts)
                step_counts_list.append(step_counts)
                break
        
        # Training
        reinforce_agent.training()
        # Recording
        scores.append(total_R)
        step_list = step_counts_list
        print("Episode: {}, Total reward: {}, Total step: {}".format(episode, total_R, step_counts_list[-1]))
    return step_counts_list
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    a = time.time()
    step_counts_list = main()
    b = time.time()
    print('Total time', b - a)
    plotPerformance(step_counts_list)