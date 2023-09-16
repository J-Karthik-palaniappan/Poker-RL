import random
from utils.cards import Card
from holdem.gameround import GameRound

class Game:
    def __init__(self,players,trainModel = False):
        print("The Poker Game begins...")
        
        self.players = players
        self.smallBlind = 10
        self.bigBlind = 20
        self.trainModel = trainModel
        self.dealerInd = 0
        
    def assignHoleCards(self):
        random.shuffle(self.deck)
        for p in self.players:
            p.holeCards = [self.deck.pop(),self.deck.pop()]
            
    def play(self):
        self.deck = Card.getDeck()
        self.assignHoleCards() #cards distribution
        openCards = []
        
        for ply in self.players:
            ply.stock = ply.stockInit
            ply.inactive = False
            print(ply, *ply.holeCards)
        
        self.round = GameRound(self.deck,self.players, self.dealerInd, self.bigBlind, self.smallBlind, self.trainModel)
        self.round.pot = 0
        
        self.round.street("preflop",openCards)
        
        openCards = [self.deck.pop(),self.deck.pop(),self.deck.pop()]
        print(*openCards)
        self.round.street("flop",openCards)
        
        openCards.append(self.deck.pop())
        print(*openCards)
        self.round.street("turn",openCards)
        
        openCards.append(self.deck.pop())
        print(*openCards)
        self.round.street("river",openCards)
        
        winPly = self.round.showDown(printResult=True)
        winPly.stock += self.round.pot
        
        self.dealerInd = (self.dealerInd+1)%len(self.players)
        
        return winPly, self.round.pot