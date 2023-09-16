import matplotlib.pyplot as plt
import numpy as np

from holdem.maingame import Game
from utils.players import RandomBot, Human
from utils.probabilities import IfElseBot
from models.dqn import DQNBot
from models.mcdqn import MCDQNBot
from models.mcdqn2 import MCDQNBot2

initial_stock = 2000
d1 = IfElseBot(initial_stock,"rule-based1")
d2 = DQNBot(initial_stock,"bot2")
d3 = MCDQNBot(initial_stock,"bot3")
d4 = MCDQNBot2(initial_stock,"bot4")
h = Human(initial_stock, "human_player")

class testing:
    def __init__(self,bot):
        self.bot = bot
        r = RandomBot(initial_stock)
        self.game = Game([bot,r],False)
        self.wins = []
        
    def play(self):
        w,a = self.game.play()
        if w==self.bot:
            self.wins.append(a)
        else:
            self.wins.append(-a)
    
    def plot(self):
        wins = np.array(self.wins)
        plt.bar(['wins','loss'],(sum(wins>0),sum(wins<0)))
        plt.title(f'{self.bot} wins and loss')
        plt.show()
        x = range(1,len(wins)+1)
        y = np.cumsum(wins)/x
        plt.plot(x,y)

########################################################################
'''
#To create a new game
g = Game([bot1,bot2,human])
g.play()

#players actions:
    fold = 0
    call_check = 1
    raise_bet = 2
    raise_twice_bet = 3
    raise_half_pot = 4
    raise_pot = 5
    allin = 6

#To train model,
g = Game([bot1,bot2],trainModel = True)
g.play()
'''
########################################################################
def training():
    game1 = Game([d2,d1])
    game2 = Game([d2,d4])
    
    t1 = testing(d2)
    t2 = testing(d4)
    
    for i in range(100): #playing 100 games
        print("-"*30,f"NEW GAME {i}","-"*30)
        game1.play()
        game2.play()
        
        if i%10 == 0:
            t1.play() #testing every 10 games
            t2.play() #testing here is playing a game with random bot
    
    t1.plot() #plotting of test results
    t2.plot()

def humanplaying():
    game = Game([h,d1,d2])
    game.play()

if __name__=='__main__':
    humanplaying()