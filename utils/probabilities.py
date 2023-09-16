import numpy as np
from utils.cards import Card,HandRankings
from utils.players import Player,Actions
import random
import copy

def simulate(deck, hole, table, nplayers):
    d1 = copy.deepcopy(deck)
    t1 = copy.deepcopy(table)
    random.shuffle(d1)
    players = [[d1.pop(),d1.pop()] for i in range(nplayers)]
    
    t1 += d1[:5-len(t1)]
    d1 = d1[5-len(t1):]
    
    myHand = Card.bestMatch(hole, t1)
    for i in range(nplayers):
        plyHand = Card.bestMatch(players[i], t1)
        if Card.compareHands(plyHand, myHand):
            return 0
    return 1

def monteCarlo(deck,hole,table,nplayers,samples=5000):
    wins = 0
    if table==[]:
        #pre calculated probability values
        return getHandStrength(hole)
    
    for i in range(samples):
        if i%500==0:
            print("monte-carlo:",i)
        wins += simulate(deck,hole,table,nplayers-1)
    print("result",wins/samples,nplayers)
    return wins/samples

#pre calculated monte carlo values for two hole cards, two players, 0 table cards
twoCardsPercentage =np.array([
    [0.85, 0.67, 0.65, 0.64, 0.64, 0.62, 0.6 , 0.59, 0.58, 0.58, 0.57, 0.56, 0.55],
    [0.65, 0.83, 0.62, 0.62, 0.61, 0.58, 0.58, 0.56, 0.55, 0.55, 0.53, 0.52, 0.52],
    [0.63, 0.61, 0.79, 0.59, 0.58, 0.56, 0.54, 0.53, 0.53, 0.51, 0.5 , 0.49, 0.48],
    [0.63, 0.6 , 0.58, 0.77, 0.56, 0.54, 0.53, 0.5 , 0.49, 0.48, 0.47, 0.46, 0.45],
    [0.61, 0.59, 0.56, 0.54, 0.75, 0.52, 0.51, 0.48, 0.47, 0.45, 0.44, 0.44, 0.43],
    [0.6 , 0.58, 0.53, 0.51, 0.5 , 0.72, 0.5 , 0.47, 0.45, 0.43, 0.41, 0.41, 0.4 ],
    [0.59, 0.54, 0.53, 0.5 , 0.48, 0.46, 0.69, 0.47, 0.44, 0.42, 0.4 , 0.39, 0.38],
    [0.58, 0.54, 0.5 , 0.48, 0.47, 0.44, 0.44, 0.66, 0.44, 0.41, 0.4 , 0.38, 0.36],
    [0.56, 0.52, 0.49, 0.46, 0.44, 0.42, 0.41, 0.39, 0.64, 0.4 , 0.38, 0.37, 0.36],
    [0.55, 0.52, 0.48, 0.46, 0.42, 0.4 , 0.39, 0.37, 0.37, 0.6 , 0.37, 0.36, 0.33],
    [0.53, 0.49, 0.47, 0.44, 0.41, 0.38, 0.37, 0.36, 0.35, 0.34, 0.56, 0.35, 0.32],
    [0.53, 0.49, 0.46, 0.43, 0.4 , 0.37, 0.35, 0.34, 0.33, 0.31, 0.3 , 0.54, 0.32],
    [0.53, 0.48, 0.45, 0.42, 0.39, 0.37, 0.34, 0.31, 0.31, 0.3 , 0.28, 0.28, 0.5 ]])

def getHandStrength(holeCards):
    r1 = 14-holeCards[0].rank
    r2 = 14-holeCards[1].rank
    a = twoCardsPercentage[r1][r2]
    b = twoCardsPercentage[r2][r1]
    if holeCards[0].suit == holeCards[1].suit:
        return max(a,b)
    return min(a,b)

class IfElseBot(Player):
    def __init__(self,cash,name=None):
        super().__init__(cash,name)
        self.prev_state = None
        self.prev_prob = None
        
    def getMove(self, env):
        moves = self.legalMoves(env)
        img = Card.cards_to_image(self.holeCards+env.openCards)
        
        if (self.prev_state is not None) and (self.prev_state==img).all():
            mc_prob = self.prev_prob
        else:
            mc_prob = monteCarlo(env.deck,
                                 self.holeCards,
                                 env.openCards,
                                 len(env.players))
        self.prev_state = img
        self.prev_prob = mc_prob
        
        print(env.betAmount, self.myCashIn)
        if mc_prob > 0.39 or env.betAmount==self.myCashIn:
            mc_prob *= np.random.uniform(0.9,1.1)*((self.stock-env.betAmount)/self.stock)
            if mc_prob > 0.85:
                return Actions.allin.value
            elif mc_prob > 0.75:
                return Actions.raise_pot.value 
            elif mc_prob > 0.65:
                return Actions.raise_half_pot.value
            elif mc_prob > 0.55:
                return Actions.raise_twice_bet.value
            elif mc_prob > 0.45:
                return Actions.raise_bet.value
            else:
                return Actions.call_check.value
        else:
            print("I am folding because my result:",mc_prob)
            return Actions.fold.value