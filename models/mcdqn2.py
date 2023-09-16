import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,concatenate,Conv2D,MaxPooling2D,Flatten

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
    cards: 4 X 13 X 1
        array([[[0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]]])
        
    winning probability:
        array([[0.87]]) #monteCarlo probability
        
    Bank:
         array([[120, #betAmount
                70, #my cash in game
                190, #current pot
                430 ]])#my stock left
'''

class MCDQNBot2(Player):
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
        
        self.path = f'models\\model_weights\\MCDQN2{MCDQNBot2.agentNum}.h5'
        if os.path.exists(self.path):
            self.learner.load_weights(self.path)
            self.target.load_weights(self.path)
        
        MCDQNBot2.agentNum += 1
        
        self.prev_state = None
        self.prev_move = None
        self.prev_reward = None
        self.history = []
        
    def build_model(self):
        inputA = Input(shape=(4,13,1)) #player's card
        inputB = Input(shape=(4,)) #bank info
        inputC = Input(shape=(1,))

        x = Conv2D(32,(3,3))(inputA) 
        x = MaxPooling2D((2,2))(x)
        x = Flatten()(x)
        x = Dense(10, activation="relu")(x)
        x = Model(inputs=inputA, outputs=x)

        y = Dense(32)(inputB) 
        y = Dense(10, activation="relu")(y)
        y = Model(inputs=inputB, outputs=y)
        
        z = Dense(1)(inputC)
        z = Model(inputs=inputC, outputs=z)

        combined = concatenate([x.output, y.output, z.output])

        p = Dense(20, activation="sigmoid")(combined)
        p = Dense(20, activation="sigmoid")(combined)
        p = Dense(self.output_size, activation="softmax")(p)

        model = Model(inputs=[x.input, y.input, z.input], outputs=p)
        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy'])
        return model
    
    def getMove(self, env):
        moves = self.legalMoves(env)
        img = Card.cards_to_image(self.holeCards+env.openCards)
        
        if (self.prev_state is not None) \
        and (np.reshape(self.prev_state[0][0],-1)==np.reshape(img,-1)).all():
            mc_prob = self.prev_state[2][0][0]
        else:
            mc_prob = monteCarlo(env.deck,
                                 self.holeCards,
                                 env.openCards,
                                 len(env.players))
        
        state = [np.expand_dims(img,axis=(0,-1)),
                 np.array([[env.betAmount,
                            self.myCashIn,
                            env.pot,
                            self.stock]]),
                 np.array([[mc_prob]])]
        
        prob = self.target.predict(state,verbose=0)
        if (np.random.uniform() < self.epsilon) and (env.trainModel):
            move = np.random.choice(moves).value
        else:
            move = np.argmax(prob)
        
        if Actions(move) not in moves:
            move = np.random.choice(moves).value
        
        if env.trainModel:
            if self.prev_state:
                self.add_transition(
                    Transition(
                        self.prev_state,
                        self.prev_move,
                        0,
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
       
    def streetOver(self,env):
        if env.current_bestPlayer == self:
            reward = env.pot
        else:
            reward = -self.myCashIn
        
        print("MY Reward:",reward)
        self.add_transition(
            Transition(
                self.prev_state,
                self.prev_move,
                reward,
                None,
                1
                )
            )
    
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
        
        current_states_cards = []
        current_states_bank = []
        current_states_prob = []
        Q_values = []
        
        for state, action, reward, next_state, done in minibatch:
            try:
                q_new = reward + self.gamma * np.amax(self.target.predict(next_state,verbose=0)[0])
            except:
                print("TRAINIG NORMALLY DIDN'T WORK!!")
                q_new = reward if reward else 0
            q = self.target.predict(state,verbose=0)
            print("old:",q)
            q[0][action] = q[0][action] + self.alpha*q_new
            print("changing:",q_new,action)
            current_states_cards.append(state[0][0])
            current_states_bank.append(state[1][0])
            current_states_prob.append(state[2][0])
            Q_values.append(q[0])
        
        current_states = [np.array(current_states_cards),np.array(current_states_bank),np.array(current_states_prob)]
        Q_values = np.array(Q_values)
        self.learner.fit(current_states, Q_values, epochs=50)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay