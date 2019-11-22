
# Initialization


```python
def createBoard():
    return np.zeros((6,7),dtype=int)
```


```python
def initializeNetworks(batchSize):
    networks = []
    for i in range(batchSize):
        
        model = models.Sequential()
        model.add(Dense(256,input_shape=(inputSize,)))
        model.add(Dense(256))
        model.add(Dense(outputSize,activation='linear'))
        model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
        
        networks.append(model)
    return networks
```

# Gameplay Logic


```python
def step(player, board, network, col):
    
    cont = True
    i = 5

    while cont == True:
        if i == -1:
            return True,board,col,player
        if board[i][col] == 0:
            board[i][col] = player
            cont = False
        else:
            i-=1
   
    return authenticate(board, player),board,col,i
```


```python
def authenticate(board, player):
    gameOver = False
    
    # search for horizontal win
    for i in range(board.shape[0]):
        row = board[i]
        for j in range(0,4):
            if [row[j],row[j+1],row[j+2],row[j+3]] == [player,player,player,player]:
                gameOver = True

    # search for vertical win
    for i in range(board.shape[1]):
        row = board.T[i]
        for j in range(0,3):
            if [row[j],row[j+1],row[j+2],row[j+3]] == [player,player,player,player]:
                gameOver = True

    # search for diagonal win
    for i in range(-2,4):
        diag = (np.diagonal(np.fliplr(board),i))
        for j in range(len(diag)-3):            
            if ([diag[j],diag[j+1],diag[j+2],diag[j+3]] == [player,player,player,player]):
                gameOver = True

    # search for diagonal win
    for i in range(-2,4):
        diag = (np.diagonal(board,i))
        for j in range(len(diag)-3):
            if ([diag[j],diag[j+1],diag[j+2],diag[j+3]] == [player,player,player,player]):
                gameOver = True
    
    return gameOver
```

# Replay Buffer


```python
def createGameplay(networks,games,counts,stateMemory,targetQ,epsilon):
    record = np.zeros(2)
    
    # games per epoch
    for i in range(games):
        board = createBoard()
        
        # stores replay memory per game
        boards = np.zeros((inputSize+1,inputSize))
        players = np.zeros((inputSize+1))
        cols = np.zeros((inputSize+1))
                
        index= 0
        win = False
        breakLoop = False
        while win == False:
            
            # player 1,2 play loop
            for player in range(1,3):
                try:
                    col = getCol(networks[player-1],board,epsilon)
                    win,board,col,i = step(player,board,networks[player-1],col)
                    
                    # tie
                    if i == -1:
                        breakLoop = True
                        break
                        
                except ValueError:
                    pass
                
                # append to replay memory
                boards[index] = board.reshape(inputSize)
                players[index] = player
                cols[index] = col
                index+=1
                
                # break game tie
                if breakLoop == True:
                    break
                
                # edit replay memory with new memories
                if win == True:
                    counts,stateMemory,targetQ = replayMemory(networks, counts, boards, players, cols, stateMemory, targetQ)
                    break

    return counts, stateMemory, targetQ
```


```python
def getCol(network,board,epsilon):
    
    # get invalid columns
    full = getFull(board)
    whileCount = 0
    
    # epsilon greedy for valid moves
    while True:
        col = epsilonGreedy(epsilon,network,board)
        if col not in full:
            break
        if whileCount == 10000:
            break
        whileCount += 1

    return col
```


```python
def epsilonGreedy(epsilon,network,board):
    # random value for epsilon-greedy
    rand = np.random.rand(1)[0]
    
    # decision based on rand
    if rand <= epsilon:
        col = np.random.randint(outputSize)
    else:
        col = np.argmax(throughNetwork(board,network))
    
    return col
```


```python
def replayMemory(networks, counts, boards, players, cols, stateMemory, targetQ):
    # for length of new memory passed in
    for i in range(len(boards)):
        agent = int(players[i])
        
        # disregard empty boards
        if sum(boards[i]) == 0:
            break
                
        # agent 1 logic
        if agent == 1:
            
            # append to memSize
            if counts//2 < memSize:
                stateMemory[agent-1][counts//2] = boards[i]
            
            # append to modulus
            else:
                stateMemory[agent-1][counts//2 % memSize] = boards[i]
        
        # agent 2 logic
        if agent == 2:
            
            # append to memSize
            if counts//2 < memSize:
                stateMemory[agent-1][counts//2] = boards[i]
            
            # append to modulus
            else:
                stateMemory[agent-1][counts//2 % memSize] = boards[i]
                
        counts+=1
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    # reset counts
    counts -= i
    
    # loop to edit targetQ's
    for i in range(len(boards)):
        agent = int(players[i])
        
        # loop for all moves in timestep t+1
        for j in range(outputSize):
        
            # disregard empty boards
            if sum(boards[i]) == 0:
                break

            # agent 1 logic
            if agent == 1:

                # append to memSize
                if counts//2 < memSize:
                    
                    # target set equal to max(next timestep) for move j
                    targetQ[agent-1][counts//2][j] = np.max(throughNetwork(
                        step(agent, np.copy(boards[i]).reshape(6,7), networks[agent-1], j)[1],networks[agent-1]))

                # append to modulus
                else:
                    
                    # target set equal to max(next timestep) for move j
                    targetQ[agent-1][counts//2 % memSize][j] = np.max(throughNetwork(
                        step(agent, np.copy(boards[i]).reshape(6,7), networks[agent-1], j)[1],networks[agent-1]))

            # agent 2 logic
            if agent == 2:

                # append to memSize
                if counts//2 < memSize:
                    
                    # target set equal to max(next timestep) for move j
                    targetQ[agent-1][counts//2][j] = np.max(throughNetwork(
                        step(agent, np.copy(boards[i]).reshape(6,7), networks[agent-1], j)[1],networks[agent-1]))

                # append to modulus
                else:
                    
                    # target set equal to max(next timestep) for move j
                    targetQ[agent-1][counts//2 % memSize][j] = np.max(throughNetwork(
                        step(agent, np.copy(boards[i]).reshape(6,7), networks[agent-1], j)[1],networks[agent-1]))
        counts+=1
    
    # append to memSize
    if counts//2 < memSize:
        
        # winning move reward
        targetQ[int(players[i-1])-1][counts//2][int(cols[i-1])] = 100
        
        # failure to block winning move
        for i in range(outputSize):
            if i != int(cols[i-1]):
                targetQ[int(players[i-2])-1][counts//2][i] = -100
        
        
    # append to modulus  
    else:
        # winning move reward
        targetQ[int(players[i-1])-1][counts//2 % memSize][int(cols[i-1])] = 100

        # failure to block winning move
        for i in range(outputSize):
            if i != int(cols[i-1]):
                targetQ[int(players[i-2])-1][counts//2 % memSize][i] = -100
    
    return counts, stateMemory, targetQ
```

# Neural Net Logic


```python
# def softmax(x):
    # exp = np.exp(x - np.max(x))
    # smax = exp / np.sum(exp)
    # return smax
```


```python
def getFull(board):
    
    # list of full/invalid columns
    full = []
    fullList = [np.sum(board.T[j] != 0) for j in range(7)]
    
    # populating full list
    for i in range(7):
        if fullList[i] == 6:
            full.append(i)
    
    return full
```


```python
def throughNetwork(board, network):
    
    # reshaping board for network input
    inputBoard = board.reshape(inputSize)
    inputBoard = board.reshape(1,-1)
    
    # getting network prediction
    output = network.predict(inputBoard,batch_size=42)
    
    return output
```


```python
def trainNetwork(networks,stateMemory,targetQ):
    
    # size of memory sample
    sampleSize = memSize//10
    
    # creating sample memory arrays
    sampleStateMemory = np.zeros((2,sampleSize,inputSize))
    sampleTargetQ = np.zeros((2,sampleSize,outputSize))
    randomSamples = np.array(random.sample(range(0,memSize),sampleSize))
    
    # populating with random samples
    for i in range(sampleSize):
        sampleStateMemory[0][i] = stateMemory[0][randomSamples[i]]
        sampleStateMemory[1][i] = stateMemory[1][randomSamples[i]]
        sampleTargetQ[0][i] = targetQ[0][randomSamples[i]]
        sampleTargetQ[1][i] = targetQ[1][randomSamples[i]]
    
    # training network
    print("Player 1 Agent...")
    networks[0].fit(sampleStateMemory[0],sampleTargetQ[0],batch_size=16,epochs=epochs)
    print("Player 2 Agent...")
    networks[1].fit(sampleStateMemory[1],sampleTargetQ[1],batch_size=16,epochs=epochs)
```

# Main Function


```python
import pandas as pd
import numpy as np
import random
import time

from keras import models
from keras import layers
from keras.layers import Dense, Activation
from keras.models import model_from_json
```


```python
inputSize = 42
outputSize = 7
memSize = 10_000
batchSize = 2

# empty state/targetQ memory
stateMemory = np.zeros((batchSize,memSize,inputSize))
targetQ = np.zeros((batchSize,memSize,outputSize))

counts = 0
```


```python
networks = initializeNetworks(batchSize)
epsilon = 1
```


```python
minEpsilon = 0.1
epsilonDecay = 0.0005

# iterations determind by epsilon decay rate
iterations = int((epsilon - minEpsilon) // epsilonDecay)
epochs = 4
games = 500

# timer
startTime = time.time()

# loop through training
for iteration in range(iterations):
    iterationStartTime = time.time()
    print("\nITERATION:", iteration,"out of",iterations)
    
    # create gameplay, epsilon decay, and training
    counts,stateMemory,targetQ = createGameplay(networks,games,counts,stateMemory,targetQ,epsilon)
    if epsilon > minEpsilon:
        epsilon-=epsilonDecay
    trainNetwork(networks,stateMemory,targetQ)
    
    # stats display every 5 iterations
    printStats(5,counts,epsilon,iterationStartTime,iteration,iterations)
    
    # after initial 500 games we only want 10 games per iteration to reduce training time
    games = 10
```


```python
def printStats(interval,counts,epsilon,iterationStartTime,iteration,iterations):
    if iteration % interval == 0:
        print("\nMemories Elapsed:",counts)
        print("Current Epsilon:",format(epsilon*100,'.2f'),"percent")
        print("\nTotal Training duration:", format((time.time() - startTime)/60,'.2f'),"mins")
        print("Current Iteration duration:", format((time.time() - iterationStartTime),'.2f'),"seconds")
        print("Estimated Time remaining:",format((time.time() - iterationStartTime)*(iterations-iteration)/60,'.2f'),"mins")
```

# Save/Load weights


```python
# Save 'em

network1_json = networks[0].to_json()
with open("network1.json", "w") as json_file:
    json_file.write(network1_json)
networks[0].save_weights("network1.h5")
print("Saved model 1 to disk")

network2_json = networks[1].to_json()
with open("network2.json", "w") as json_file:
    json_file.write(network2_json)
networks[1].save_weights("network2.h5")
print("Saved model 2 to disk")

# SOURCE CODE: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
```


```python
# Load 'em

json_file = open('network1.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("network1.h5")
networks[0] = loaded_model
print("Loaded Model from disk")

json_file = open('network2.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("network2.h5")
networks[1] = loaded_model
print("Loaded Model from disk")

# SOURCE CODE: https://machinelearningmastery.com/save-load-keras-deep-learning-models/

networks[0].compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
networks[1].compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
```


```python
# Save memories

pd.DataFrame(stateMemory[0]).to_csv("stateMemory1.csv",index=False)
pd.DataFrame(stateMemory[1]).to_csv("stateMemory2.csv",index=False)
pd.DataFrame(targetQ[0]).to_csv("targetQ1.csv",index=False)
pd.DataFrame(targetQ[1]).to_csv("targetQ2.csv",index=False)
pd.DataFrame(np.array(counts).reshape(1,1)).to_csv("memCounts.csv",index=False)
```


```python
# import memories
import pandas as pd

stateMemory[0] = pd.read_csv("stateMemory1.csv").values
stateMemory[1] = pd.read_csv("stateMemory2.csv").values
targetQ[0] = pd.read_csv("targetQ1.csv")
targetQ[1] = pd.read_csv("targetQ2.csv")
counts = pd.read_csv("memCounts.csv").values[0][0]
```

# TODO
- fix output cols
- fix loser reward
