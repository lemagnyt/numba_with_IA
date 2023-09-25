import numpy as np
import random
import time
import numba
import math
from numba import jit  # jit convertit une fonction python => fonction C
from tqdm import tqdm
###################################################################

# PLayer 0 => Vertical    Player
# PLayer 1 => Horizontal  Player

# IdMove : code servant à identifier un coup particulier sur le jeu
# P   : id player 0/1
# x,y : coordonnées de la tuile, Player0 joue sur (x,y)+(x,y+1) et Player1 sur (x,y)+(x+1,y)

# convert: player,x,y <=> IDmove

# IDmove=123 <=> player 1 plays at position x = 2 and y = 3
# ce codage tient sur 8 bits !

@jit(nopython=True)
def GetIDmove(player,x,y):
    return player * 100 + x * 10 + y

@jit(nopython=True)
def DecodeIDmove(IDmove):
    y = IDmove % 10
    x = int(IDmove/10) % 10
    player = int(IDmove / 100)
    return player,x,y

###################################################################

# Numba requiert des numpy array pour fonctionner

# toutes les données du jeu sont donc stockées dans 1 seul array numpy

# Data Structure  - numpy array de taille 144 uint8 :
# B[ 0- 63] List of possibles moves
# B[64-127] Gameboard (x,y) => 64 + x + 8*y
# B[-1] : number of possible moves
# B[-2] : reserved
# B[-3] : current player




StartingBoard  = np.zeros(144,dtype=np.uint8)

@jit(nopython=True)   # pour x,y donné => retourne indice dans le tableau B
def iPxy(x,y):
    return 64 + 8 * y + x

@jit(nopython=True)
def _PossibleMoves(idPlayer,B):   # analyse B => liste des coups possibles par ordre croissant
    nb = 0

    #player V
    if idPlayer == 0 :
        for x in range(8):
            for y in range(7):
                p = iPxy(x,y)
                if B[p] == 0 and B[p+8] == 0 :
                    B[nb] = GetIDmove(0,x,y)
                    nb+=1
    # player H
    if idPlayer == 1 :
        for x in range(7):
            for y in range(8):
                p = iPxy(x,y)
                if B[p] == 0 and B[p+1] == 0 :
                    B[nb] = GetIDmove(1,x,y)
                    nb+=1

    B[-1] = nb

_PossibleMoves(0,StartingBoard)   # prépare le gameboard de démarrage


###################################################################

# Numba ne gère pas les classes...

# fonctions de gestion d'une partie
# les fonctions sans @jit ne sont pas accélérées

# Player 0 win => Score :  1
# Player 1 win => Score : -1


# def CreateNewGame()   => StartingBoard.copy()
# def CopyGame(B)       => return B.copy()

@jit(nopython=True)
def Terminated(B):
    return B[-1] == 0

@jit(nopython=True)
def GetScore(B):
    if B[-2] == 10 : return  1
    if B[-2] == 20 : return -1
    return 0


@jit(nopython=True)
def Play(B,idMove):
    player,x,y = DecodeIDmove(idMove)
    p = iPxy(x,y)

    B[p]   = 1
    if player == 0 : B[p+8] = 1
    else :           B[p+1] = 1

    nextPlayer = 1 - player

    _PossibleMoves(nextPlayer,B)
    B[-3] = nextPlayer

    if B[-1] == 0  :             # gameover
        B[-2] = (player+1)*10    # player 0 win => 10  / player 1 win => 20


@jit(nopython=True)
def Playout(B):
    while B[-1] != 0:                   # tant qu'il reste des coups possibles
        id = random.randint(0,B[-1]-1)  # select random move
        idMove = B[id]
        Play(B,idMove)


##################################################################
#
#   for demo only - do not use for computation

def Print(B):
    for yy in range(8):
        y = 7 - yy
        s = str(y)
        for x in range(8):
            if     B[iPxy(x,y)] == 1 : s += '::'
            else:                      s += '[]'
        print(s)
    s = ' '
    for x in range(8): s += str(x)+str(x)
    print(s)


    nbMoves = B[-1]
    print("Possible moves :", nbMoves);
    s = ''
    for i in range(nbMoves):
        s += str(B[i]) + ' '
    print(s)

@numba.jit(nopython=True, parallel=True)
def ParrallelPlayout(nb,B1):
    Scores = np.empty(nb)
    for i in numba.prange(nb):
        BCopy = B1.copy()
        Playout(BCopy)
        Scores[i] = GetScore(BCopy)
    return Scores.mean()

def IARandom(B1):
    id = random.randint(0,B1[-1]-1)
    return B1[id]

def IA100P(B1):
    return Simulates(B1,100)

def IA1000P(B1):
    return Simulates(B1,1000)

def IA10000P(B1):
    return Simulates(B1,10000)

def Simulates(B1,nbSimus,MCTS=False):
    player = B1[-3]
    results = {}
    for idMove in B1[0:B1[-1]]:
        B2 = B1.copy()
        Play(B2,idMove)
        results[idMove]=ParrallelPlayout(nbSimus,B2)
    if not MCTS :
        if player==0 :    #car score gagnant = 1
            return max(results, key=lambda key: results[key])
        else :            #car score gagnant = -1
            return min(results, key=lambda key: results[key])
    else :
        return results;

def PvP(B1,IA0,IA1):
    while B1[-1] != 0:                   # tant qu'il reste des coups possibles
        if B1[-3]== 0 :
            Play(B1,IA0(B1))
        else :
            Play(B1,IA1(B1))

def PvPDebug(B1,IA0,IA1):
    while B1[-1] != 0:                   # tant qu'il reste des coups possibles
        if B1[-3]== 0 :
            IAMove = IA0(B1)
        else :
            IAMove = IA1(B1)
        player,x,y = DecodeIDmove(IAMove)
        print("Playing : ",IAMove, " -  Player: ",player, "  X:",x," Y:",y)
        Play(B1,IAMove)
        Print(B1)
        print("---------------------------------------")


def PlayoutDebug(B,verbose=False):
    Print(B)
    while not Terminated(B):
        id = random.randint(0,B[-1]-1)  # select random move
        idMove = B[id]
        player,x,y = DecodeIDmove(idMove)
        print("Playing : ",idMove, " -  Player: ",player, "  X:",x," Y:",y)
        Play(B,idMove)
        Print(B)
        print("---------------------------------------")


def PvPSimu(B1,player1,player2,nbSimus,debug=False):
    WinP0 = 0
    for i in tqdm(range(0,nbSimus)):
        BCopy = B1.copy()
        if debug :
            PvPDebug(BCopy,player1,player2)
        else :
            PvP(BCopy,player1,player2)
        #print("\nScore : ",GetScore(BCopy))
        if GetScore(BCopy)==1 :
            WinP0+=1
        elif GetScore(BCopy)!= -1 :
            print('Error')
            return
    WinPorcentP0 = int(WinP0*100/nbSimus)
    print(str(WinPorcentP0)+'% IA0 - '+str(100-WinPorcentP0)+'% IA1')

def main_pvp():
    B = StartingBoard.copy()
    PvPSimu(B,IARandom,IA100P,10)
    PvPSimu(B,IA100P,IA100P,10)
    PvPSimu(B,IA100P,IA1000P,10)
    PvPSimu(B,IA1000P,IA1000P,10)
    PvPSimu(B,IA1000P,IA10000P,10)
    PvPSimu(B,IA10000P,IA10000P,10)
   
def UCB(coup,player):
    if(player==0):  
        return coup['mean']+math.sqrt(math.log(coup['parent']['n'])/coup['n']) 
    return -coup['mean']+math.sqrt(math.log(coup['parent']['n'])/coup['n']) 
        
def MCTS(B1,deltaTime=2):   
    #Pour l'initialisation, on fait simulation et expansion
    #Simulates correspond au tableau des moyennes de score pour chaque coup possible dans cet état lorsque l'on met MCTS à True
    firstSimu = Simulates(B1,1000,MCTS=True)
    main = {'childrens':{},'parent':None,'B':B1.copy(),'mean':0,'n':0}
    for coup in B1[0:B1[-1]]:
        main['childrens'][coup]={'childrens':{},'n':1000,'mean':firstSimu[coup],'parent':main}
        main['mean']=(main['mean']*main['n']+firstSimu[coup]*1000)/(main['n']+1000)
        main['n']+=1000
        B2 = B1.copy()
        Play(B2,coup)
        main['childrens'][coup]['B']=B2
    startTime = time.time()
    currentState=main
    while((time.time()-startTime < deltaTime) and currentState['B'][-1]!=0):
        #SELECTION
        currentState=main
        while(len(currentState['childrens'])):
            player = currentState['B'][-3] 
            currentState = currentState['childrens'][max(currentState['childrens'], key=lambda key:UCB(currentState['childrens'][key],player))]
        #EXPANSION
        for coup in currentState['B'][0:currentState['B'][-1]]:
            currentState['childrens'][coup]={'childrens':{},'parent':currentState}
            B2 = currentState['B'].copy()
            Play(B2,coup)
            currentState['childrens'][coup]['B']=B2
        #SIMULATION
        currentSimu =  Simulates(currentState['B'],1000,MCTS=True)
        for coup in currentState['childrens']:
            currentState['childrens'][coup]['mean']=currentSimu[coup]
            currentState['childrens'][coup]['n']=1000
            #BACKPROPAGATION
            childrenState = currentState['childrens'][coup]
            while(childrenState['parent']):
                parentState = childrenState['parent']
                parentState['mean']=(parentState['mean']*parentState['n']+currentSimu[coup]*1000)/(parentState['n']+1000)
                parentState['n']+=1000
                childrenState = parentState
    if B1[-3]==0 :
        return max(main['childrens'], key=lambda key: main['childrens'][key]['mean'])
    return min(main['childrens'], key=lambda key: main['childrens'][key]['mean'])
                
            
        
        
        
        

################################################################
#
#  Version Debug Demo pour affichage et test

B = StartingBoard.copy()
PvPSimu(B,MCTS,IA10000P,10,debug=True)
#main_pvp()
# B = StartingBoard.copy()
# MCTS(B,2)

################################################################
#
#   utilisation de numba => 100 000 parties par seconde

# print("Test perf Numba")

# T0 = time.time()
# nbSimus = 0
# while time.time()-T0 < 2:
#     B = StartingBoard.copy()
#     Playout(B)
#     nbSimus+=1
# print("Nb Sims / second:",nbSimus/2)


################################################################
#
#   utilisation de numba +  multiprocess => 1 000 000 parties par seconde

# print()
# print("Test perf Numba + parallélisme")





# nbSimus = 10 * 1000 * 1000
# T0 = time.time()
# MeanScores = ParrallelPlayout(nbSimus,StartingBoard)
# T1 = time.time()
# dt = T1-T0

# print("Nb Sims / second:", int(nbSimus / dt ))














