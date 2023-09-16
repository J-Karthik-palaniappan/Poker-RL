from itertools import combinations
import enum
import numpy as np
import random

class HandRankings(enum.Enum):
    StraightFlush = 1
    FourOfAKind = 2
    FullHouse = 3
    Flush = 4
    Straight = 5
    ThreeOfAKind = 6
    TwoPair = 7
    OnePair = 8
    HighCard = 9
    
    def __str__(self):
        return self._name_
    
class Card:
    suits = ('spade','heart','club','dice')
    ranks = (2,3,4,5,6,7,8,9,10,11,12,13,14)
    picture = {'heart':"\u2665",'spade':"\u2660",'dice':"\u2666",'club':"\u2663"}
    
    def __init__(self,rank,suit):
        self.rank = rank
        self.suit = suit
        self.name = str(self.rank)
        if self.rank == 10:
            self.name = 'T'
        elif self.rank == 11:
            self.name = 'J'
        elif self.rank == 12:
            self.name = 'Q'
        elif self.rank == 13:
            self.name = 'K'
        elif self.rank == 14:
            self.name = 'A'
    
    def __str__(self):
        return self.name+Card.picture[self.suit]
    
    def getDeck():
        d = [Card(j,i) for i in Card.suits for j in Card.ranks]
        random.shuffle(d)
        return d
    
    def bestMatch(holecards,opencards):
        '''
        Parameters
        ----------
        holecards : 2 cards of player
            [A-spade, 5-club]
        opencards : community cards (3 to 5)
            [3-spade, 4-dice, 7-spade, 6-heart, A-heart]

        Returns
        -------
        bestHand : {Cards: 5 cards, HandRank: handrank} 
        '''
        bestHand = None
        for comb in combinations(opencards+holecards,5):
            cards = list(comb)
            myHand = {'Cards':cards,
                      'HandRank':Card.getHandRank(cards)}

            if Card.compareHands(myHand,bestHand):
                bestHand = myHand

        return bestHand
    
    def getHandRank(cards):
        s = Card.isStraight(cards)
        f = Card.isFlush(cards)
        if s and f:
            HandRank = HandRankings.StraightFlush
        else:
            HandRank = Card.allPairs(cards)
            if HandRank.value > 3:
                if f:
                    HandRank = HandRankings.Flush
                elif s:
                    HandRank = HandRankings.Straight
        return HandRank
            
    def isStraight(cards):
        string = ''
        c1 = sorted(cards,key = lambda x:x.rank)
        for i in c1:
            string += i.name
        if string in 'A23456789TJQKA':
            return True
        return False
    
    def isFlush(cards):
        prev = cards[0].suit
        for i in cards:
            if i.suit!=prev:
                return False
        return True
    
    def allPairs(cards):
        pairs = {}
        for i in cards:
            pairs[i.rank] = pairs.get(i.rank,0) + 1
        pairs = list(pairs.values())
        twos = pairs.count(2)
        threes = pairs.count(3)
        fours = pairs.count(4)
        if fours:
            return HandRankings.FourOfAKind
        elif twos == 1 and threes == 1:
            return HandRankings.FullHouse
        elif threes == 1:
            return HandRankings.ThreeOfAKind
        elif twos == 2:
            return HandRankings.TwoPair
        elif twos == 1:
            return HandRankings.OnePair
        else:
            return HandRankings.HighCard
        
    def compareHands(hand1,hand2):
        '''
        hand input: {'Cards':[4-spade,5-heart,...],'HandRank':HandRankings.OnePair}
        returns True if hand1 better than hand2
        '''
        if not hand2:
            return True
        elif not hand1:
            return False
        elif hand1['HandRank'].value < hand2['HandRank'].value:
            return True
        elif hand1['HandRank'].value > hand2['HandRank'].value:
            return False
        else:
            c1 = [i.rank for i in hand1['Cards']]
            c2 = [i.rank for i in hand2['Cards']]
            c1 = sorted(c1,key = lambda x:x + c1.count(x)*14,reverse=True)
            c2 = sorted(c2,key = lambda x:x + c2.count(x)*14,reverse=True)
            for i,j in zip(c1,c2):
                if i>j:
                    return True
                elif i<j:
                    return False
            return True
        
    def cards_to_image(cards):
        arr = np.zeros((4,13))
        suit_order = ['spade','heart','club','dice']
        for i in cards:
            r = i.rank-2
            s = suit_order.index(i.suit)
            arr[s][r] = 1        
        
        return arr