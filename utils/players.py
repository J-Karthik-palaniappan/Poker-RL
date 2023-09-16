import numpy as np
import enum

class Actions(enum.Enum):
    fold = 0
    call_check = 1
    raise_bet = 2
    raise_twice_bet = 3
    raise_half_pot = 4
    raise_pot = 5
    allin = 6
    
    def __str__(self):
        return self._name_

class Player:
    playerNum = 0
    def __init__(self,stock,name=None):
        self.stockInit = stock
        self.stock = stock
        self.name = name if name else f'player{Player.playerNum}'
        Player.playerNum += 1
        self.holeCards = None
        self.myCashIn = 0
        self.inactive = False
        self.moved = False
        
    def __str__(self):
        return self.name + " cash:" + str(self.stock)
        
    def bet(self,amount):
        if amount <= self.stock:
            self.stock -= amount
            self.myCashIn += amount
            return amount
        elif self.stock > 0:
            c,self.stock = self.stock,0
            self.myCashIn += c
            return c
        else:
            print(f"{self} is broke!")
            self.inactive = True
            return 0
        
    def legalMoves(self, env):
        moves = list(Actions)
        callCash = env.betAmount - self.myCashIn
        inc = env.betAmount if env.betAmount else env.bigBlind
        if self.stock < env.pot + callCash:
            moves.remove(Actions.raise_pot)
        if self.stock < env.pot//2 + callCash:
            moves.remove(Actions.raise_half_pot)
        if self.stock < inc*2 + callCash:
            moves.remove(Actions.raise_twice_bet)
        if self.stock < inc + callCash:
            moves.remove(Actions.raise_bet)
        if self.stock < callCash:
            moves.remove(Actions.call_check)
        if self.stock == 0:
            moves.remove(Actions.allin)

        return moves
    
    def setReward(self,reward):
        pass
    def streetOver(self,env):
        pass
    
class Human(Player):
    def __init__(self,cash,name=None):
        super().__init__(cash,name)
        
    def getMove(self, env):
        moves = self.legalMoves(env)
        move = int(input(f"{self}:"))
        if Actions(move) not in moves:
            move = self.getMove(env)
        return move
    
class RandomBot(Player):
    def __init__(self,cash,name=None):
        super().__init__(cash,name)
        
    def getMove(self, env):
        moves = self.legalMoves(env)
        move = np.random.choice(moves)
        return move