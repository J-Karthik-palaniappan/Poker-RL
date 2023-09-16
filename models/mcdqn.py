import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense

from collections import deque, namedtuple
import numpy as np
import random
import os

from utils.players import Player, Actions
from utils.cards import Card
from utils.probabilities import monteCarlo

Transition = namedtuple('Transition', 'state, action, reward, next_state, done')
'''
State of the model is a list cards, probability and bank info:
    cards: 57 X 1
        array([[0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, #card
                0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, #card
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, #card
                0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0] #card
                0.87, #monteCarlo probability
                120, #betAmount
                70, #my cash in game
                190, #current pot
                430 #my stock left
                ]])
'''

class MCDQNBot(Player):
    agentNum = 0
    def __init__(self, cash, name=None):
        super().__init__(cash,name)
        self.output_size = 7
        
        self.alpha = 0.7
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.01
        self.batch_size = 32
        self.copy_every = 150
        
        self.memory = deque(maxlen=5000)
        self.timestep = 0
        
        self.learner = self.build_model()
        self.target = self.build_model()
        
        self.path = f'models\\model_weights\\MCDQN{MCDQNBot.agentNum}.h5'
        if os.path.exists(self.path):
            self.learner.load_weights(self.path)
            self.target.load_weights(self.path)
        
        MCDQNBot.agentNum += 1
        
        self.prev_state = None
        self.prev_move = None
        self.prev_reward = None
        self.history = []
        
    def build_model(self):
        inp = Input(shape=(57,))

        y = Dense(64, activation="sigmoid")(inp) 
        y = Dense(128, activation="relu")(y)
        y = Dense(self.output_size, activation="sigmoid")(y)
        model = Model(inputs=inp, outputs=y)

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy'])
        return model
    
    def getMove(self, env):
        moves = self.legalMoves(env)
        img = Card.cards_to_image(self.holeCards+env.openCards)
        img = np.reshape(img,-1)
        
        if (self.prev_state is not None) and (self.prev_state[0,:52]==img).all():
            mc_prob = self.prev_state[0,52]
        else:
            mc_prob = monteCarlo(env.deck,
                                 self.holeCards,
                                 env.openCards,
                                 len(env.players))
        
        state = np.concatenate((img,[mc_prob,
                                     env.betAmount,
                                     self.myCashIn,
                                     env.pot,
                                     self.stock]))
        state = np.expand_dims(state,axis=0)
        
        prob = self.target.predict(state,verbose=0)
        if (np.random.uniform() < self.epsilon) and (env.trainModel):
            move = np.random.choice(moves).value
        else:
            move = np.argmax(prob)
        
        if Actions(move) not in moves:
            move = np.random.choice(moves).value
        
        if env.trainModel:
            if self.prev_state is not None:
                self.add_transition(
                    Transition(
                        self.prev_state,
                        self.prev_move,
                        self.prev_reward,
                        state,
                        0
                        )
                    )
            self.prev_state = state
            self.prev_move = move
        self.history.append(move)

        return move
    
    def setReward(self,reward):
       self.prev_reward = reward
    
    def add_transition(self, transition):
        self.memory.append(transition)
        self.timestep += 1

        if self.timestep >= 100 and self.timestep % 50 == 0:
            self.replay()
            self.learner.save_weights(self.path)

        if self.timestep % self.copy_every == 0:
            self.target.set_weights(self.learner.weights)
            print(f'target parameters updated on step : {self.timestep}')
            
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        
        current_states = []
        Q_values = []
        
        for state, action, reward, next_state, done in minibatch:
            try:
                q_new = reward + self.gamma * np.amax(self.target.predict(next_state,verbose=0)[0])
            except:
                print("TRAINIG NORMALLY DIDN'T WORK!!")
                q_new = reward if reward else 0
            q = self.target.predict(state,verbose=0)
            q[0][action] = q[0][action] + self.alpha*q_new
            
            current_states.append(state[0])
            Q_values.append(q[0])
        
        current_states = np.array(current_states)
        Q_values = np.array(Q_values)
        self.learner.fit(current_states, Q_values, epochs=50)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay