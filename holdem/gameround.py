from utils.players import Actions, Human
from utils.probabilities import getHandStrength
from utils.cards import Card

class GameRound:
    def __init__(self,
                 deck,
                 players,
                 dealerInd,
                 bigBlind,
                 smallBlind,
                 trainModel):
        
        self.deck = deck
        self.players = players
        self.dealerInd = dealerInd #dealer's (first) index
        self.smallBlind = smallBlind
        self.bigBlind = bigBlind
        self.openCards = None
        self.current_bestPlayer = None
        self.trainModel = trainModel
        self.pot = 0
        
    def setup(self):
        self.betAmount = 0
        self.ind = self.dealerInd
        for i in self.players:
            i.moved = False
            i.myCashIn = 0
        
    def blinds(self):
        print(f"Dealer: {self.players[self.dealerInd]}")
        
        p = self.nextPlayer()
        amount = p.bet(self.smallBlind)
        print(f"SmallBlind: {p} amount: {amount}")
        self.pot += amount
        
        p = self.nextPlayer()
        amount = p.bet(self.bigBlind)
        print(f"BigBlind: {p} amount: {amount}")
        self.pot += amount
        self.betAmount = self.bigBlind
        
    def street(self, roundName, openCards):
        print(f"================={roundName}=================")
        
        self.openCards = openCards
        self.setup()
        self.current_bestPlayer = self.showDown()
        if roundName == "preflop":
            self.blinds()

        while(not self.hasTerminated()):
            p = self.nextPlayer()
            p.moved = True
            
            print(f'pot:{self.pot} bet:{self.betAmount}')
            move = p.getMove(self)#getting move and training
            if not isinstance(p,Human):
                print(f'🤖 {p}:{Actions(move)}')
            
            if Actions(move) == Actions.fold:
                p.inactive = True
                continue
            
            callCash = self.betAmount - p.myCashIn
            inc = self.betAmount if self.betAmount else self.bigBlind
            
            if Actions(move) == Actions.call_check:
                amount = p.bet(callCash)
                
            elif Actions(move) == Actions.raise_bet:
                amount = p.bet(callCash + inc)
                
            elif Actions(move) == Actions.raise_twice_bet:
                amount = p.bet(callCash + inc*2)
                    
            elif Actions(move) == Actions.raise_half_pot:
                amount = p.bet(callCash + self.pot//2)
                
            elif Actions(move) == Actions.raise_pot:
                amount = p.bet(callCash + self.pot)
                
            elif Actions(move) == Actions.allin:
                amount = p.bet(p.stock)
                
            else:
                raise Exception('wrong Move!')
            
            if p.myCashIn > self.betAmount:
                self.betAmount = p.myCashIn
                
            self.pot += amount
            p.setReward(self.playerReward(p, amount))
            #print(p,"*OLD REWARD:*\b",self.playerReward(p, amount))
        
        print(f"-----{roundName} over-----")
        
    def hasTerminated(self):
        notPlayed = 0
        inactives = 0
        for ply in self.players:
            if ply.inactive or (ply.stock==0):
                inactives += 1
                continue
            if ply.myCashIn < self.betAmount:
                return False
            if not ply.moved:
                notPlayed += 1
        
        if notPlayed > 1: #if two have not played
            return False
        elif notPlayed == 1 and inactives < len(self.players)-1:#if one has not played, check if all others are inactive
            return False
        
        for ply in self.players:
            ply.streetOver(self)
        return True
        
    def nextPlayer(self,loop = 0):
        self.ind = (self.ind+1)%len(self.players)
        ply = self.players[self.ind]
        if loop > len(self.players):
            raise Exception("no player has cash - infinite loop")
        elif (ply.inactive) or (ply.stock == 0):
            loop += 1
            ply = self.nextPlayer(loop)
        return ply
    
    def playerReward(self,player,amount): 
        if player == self.current_bestPlayer:
            return amount
        return -amount

    def showDown(self,printResult = False):
        if not self.openCards:
            winPly = self.players[0]
            for ply in self.players:
                if getHandStrength(winPly.holeCards) < getHandStrength(ply.holeCards):
                    winPly = ply
            return winPly
        
        winPly = None 
        winHand = None
        
        for ply in self.players:
            if ply.inactive:
                continue
            myHand = Card.bestMatch(ply.holeCards, self.openCards)
            if printResult:print(ply,"hole:",*ply.holeCards,"match:",*myHand['Cards'],myHand['HandRank'])
            
            if Card.compareHands(myHand, winHand):#if myHand better than winHand
                winHand = myHand
                winPly = ply
                
        if printResult:   
            print(30*"*",winPly.name,"won",self.pot,"*"*30)
            print("winner:",winPly,*winHand['Cards'],winHand['HandRank'])
        
        #if all folded, dealer wins
        return winPly if winPly else self.players[self.dealerInd]