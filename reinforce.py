
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
FIG_FILE = 'Lr={}, GAMMA={}, time={}_.png'.format(LR, GAMMA, datetime.now().strftime("%m%d%Y%H%M%S"))

def handler(signum, frame):
    msg = "Ctrl-C was pressed. Screenshot was saved at the current local folder. \n"
    print()
    print(msg, end="", flush=True)
    plotPerformance(avg_list, step_list)
    sys.exit(1)

def plotPerformance(avg_loss_list, step_counter_list):
    cut = 0
    for i in range(len(avg_loss_list)):
        if not math.isnan(avg_loss_list[i]):
            # print(i)
            cut = i
            break
    step_counter_list = step_counter_list[cut:]
    avg_loss_list = avg_loss_list[cut:]
    x = np.arange(0,len(step_counter_list))
    y1 = step_counter_list
    y2 = avg_loss_list

    if len(y1) <= 300:
        window_len = 5
    else:
        51
    y1_smooth = savgol_filter(y1, window_len, 3)
    y2_smooth = savgol_filter(y2, window_len, 3)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y1_smooth, 'g-',label='total steps')
    ax2.plot(x, y2_smooth, 'b--',label='average loss')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.set_xlabel('Training episode')
    ax1.set_ylabel('total steps')
    ax2.set_ylabel('average loss')
    plt.savefig(FIG_FILE)

class PolicyNet():
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
        nn = PolicyNet(N_STATES, N_ACTIONS)
        # self.model = MakeModel(2)
        self.model = nn.net_fc()
        self.S_buffer = []
        self.A_buffer = []
        self.R_buffer = []
        self.optim = OPTIM
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
        for i in range(len(episode_R)):
            return_G = 0
            pow = 0
            for R in episode_R[i:]:
                return_G += R * GAMMA ** pow
                pow += 1
            returns_G.append(return_G)
        # returns_G = (returns_G - np.mean(returns_G)) / (np.std(returns_G) + 1e-10) # normilization
        return returns_G # R1, R2, ... , Rt.
    
    def calLoss(self, S, A, R): 
        probs = self.model(S)[0]
        log_prob = tf.math.log(probs[A])
        loss = -log_prob * R
        return loss 

    def training(self):
        returns_G = self.calReturns()
        losses = []
        for S, A, R in zip(self.S_buffer, self.A_buffer, returns_G):
            with GradientTape() as gt:
                loss = self.calLoss(S, A, R)
                grads = gt.gradient(loss, self.model.trainable_variables)
                OPTIM.apply_gradients(zip(grads, self.model.trainable_variables))
                losses.append(loss)

        self.S_buffer = []
        self.A_buffer = []
        self.R_buffer = []
        return np.mean(losses)

def main():
    global avg_list # For plotting
    global step_list
    ENV = gym.make(ENVNAME)  # make game env
    N_STATES = ENV.observation_space.shape[0] # 4
    N_ACTIONS = ENV.action_space.n # 2
    reinforce_agent = ReinforceAgent(N_STATES, N_ACTIONS)
    step_counts_list = [] # for recording the number of the step the agent do in each game
    scores = [] # for recording the cumulative rewards in each game
    avg_loss_list = [] # for recording the change in loss when training
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
            reinforce_agent.appendBuffer(S, A, R) # Replay buffer
            S = S_next
            total_R += R

            # End if there are enough winnings.
            if terminal:
                # Dqn_agent.target_model.set_weights(Dqn_agent.original_model.get_weights()) # Update target model every episode
                win_most_recent.append(step_counts)
                step_counts_list.append(step_counts)
                break
        
        # Training
        avg_losses = reinforce_agent.training()
        # Recording
        scores.append(total_R)
        avg_loss_list.append(avg_losses)
        avg_list = avg_loss_list
        step_list = step_counts_list
        print("Episode: {}, Total reward: {}, Total step: {}".format(episode, total_R, step_counts_list[-1]))
    # print('Scores: ', scores)
    # print('Steps: ', step_counts_list)
    return avg_loss_list, step_counts_list
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    a = time.time()
    avg_loss_list, step_counts_list = main()
    b = time.time()
    print('Total time', b - a)
    plotPerformance(avg_loss_list, step_counts_list)