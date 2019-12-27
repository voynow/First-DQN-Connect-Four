
# Intelligent Connect Four with Deep Q-Learning
### Created by: Jamie Voynow<br>

The following section of code manifests an agent. This agent was tasked with learning to play the game of Connect Four without explicit instructions. Below we use deep q-learning (a method of reinforcement learning) to obtain this game-play knowledge. Q learning has a few distinct characteristics that make it effective on certain RL environments. These characteristics will be highlighted and explained in depth below.<br><br>

# Initialization


```python
def createBoard():
    
    # returns numpy array of empty board
    return np.zeros((6,7),dtype=int)
```


```python
def initializeNetworks(batchSize):
    
    # loop to populate agents into networks list
    networks = []
    for i in range(batchSize):
        
        # keras Sequential model arcitecture
        model = models.Sequential()
        model.add(Dense(256,input_shape=(inputSize,)))
        model.add(Dense(256))
        model.add(Dense(outputSize,activation='linear'))
        model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
        
        networks.append(model)
    return networks
```

# Gameplay Logic<br>

The following two functions are associated with the functionality of the connect four game. The step function is in place in order to make a move given a player, board, and column. The authenticate function deals with checking the game board for winning states.<br><br>


```python
def step(player, board, col):
    
    cont = True
    i = 5
    
    while cont == True:
        if i == -7:
            return False, board, col, i
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
The replay buffer serves as a memory of game-play for a given agent. This memory consists of a mapping from states to their associated value. The following two functions are associated with the creation of these replays and do so in accessing the game-play functions. The third function, replayMemory, is where the replay buffer is manifested.
<br><br>
The first function, createGameplay, loops through our specified number of games to play per iteration. In each loop the agents play their game of connect four until there is a winner. While this game-play is happening, we collect that data from the games (states and rewards) which will be used as replay memory.
<br><br>
The second function, epsilonGreedy, is called to determine which column an agent believes will give the highest reward. The value of epsilon (between 0 and 1) is passed into this function along with the agent's network and current game board. This epsilon value is used to determine the amount of random moves an agent takes. This is a simple solution to the exploration vs exploitation problem, and we call this solution epsilon-greedy. This epsilon value decays over iterations. This decay rate is determined by our epsilonDecay value located in Main.
<br><br>
The third function, replayMemory, is called after one iteration of game-play is over and is passed the new data from that iteration of game-play. This function's purpose is to create this replay memory which is made of states and rewards. The states are passed into the function which are then seperated by agent into their appropriate list for storage. In order to determine the value of being in a specific state, we do a forward pass from a given state and take the max value from this new state and its potential moves. This is how we approximate our value function. These max values are then stored as target Q values which we will later train our network on. The states and their associated target values are then stored in our memory list. <br><br>


```python
def createGameplay(networks,games,memoryCount,memory,epsilon):
    
    # games per epoch
    for i in range(games):
        board = createBoard()
        
        # stores replay memory per game
        boards = np.zeros((inputSize+1,inputSize))
        players = np.zeros((inputSize+1))
        cols = np.zeros((inputSize+1))
                
        index= 0
        win = False
        tieGame = False
        while win == False:
            
            # player 1,2 play loop
            for player in range(1,3):
                try:
                    col = epsilonGreedy(epsilon,networks[player-1],board)
                    win,board,col,i = step(player,board,col)
                    
                    # tie
                    if col == -1 or i == -7:
                        tieGame = True
                        break
                        
                except ValueError:
                    pass
                
                # append to replay memory
                boards[index] = board.reshape(inputSize)
                players[index] = player
                cols[index] = col
                index+=1
                
                # edit replay memory with new memories
                if win == True:
                    gameMemory = replayMemory(networks, boards, players, cols)
                    if memoryCount < memSize:
                        memory.append(gameMemory)
                    else:
                        memory[memoryCount % memSize] = gameMemory
                    memoryCount+=1
                    
                    break
            
            if tieGame == True:
                break

    return memoryCount, memory
```


```python
def epsilonGreedy(epsilon,network,board):
    # random value for epsilon-greedy
    rand = np.random.rand(1)[0]
    
    # decision based on rand
    if rand <= epsilon:
        
        # check if board is full
        full = getFull(board)
        if len(full) == outputSize:
            return -1
        
        # only allow moves to available columns
        available = []
        for i in range(outputSize):
            if i not in full:
                available.append(i)

        col = random.choice(available)
        
    else:
        
        # take position associated with the max reward
        col = np.argmax(throughNetwork(board,network))
    
    return col
```


```python
def replayMemory(networks, boards, players, cols):
    
    # board lists
    boardsA = []
    boardsB = []
    
    # counters
    countA = 0
    countB = 0
    
    # loop through all valid boards
    i = 0
    while np.sum(boards[i]) != 0:
        if players[i] == 1:
            boardsA.append(boards[i])
            countA+=1
            
        if players[i] == 2:
            boardsB.append(boards[i])
            countB+=1
            
        i+=1
    
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  # 
    
    # creating empty targetQ np array
    tempTargetQ = np.zeros((i,outputSize))
    
    # loop for same length as boards loop
    for j in range(i-1):
        full = getFull(boards[j].reshape(6,7))
        for k in range(outputSize):
            
            # we teach the agent to make valid moves by setting the reward associated with invalid moves as 0
            if k in full:
                tempTargetQ[j][k] = 0
                
            else:
                # we take the max value of our forward pass after simulating one timestep into the future
                tempTargetQ[j][k] = np.max(throughNetwork(
                    step(int(players[j]), np.copy(boards[j+1]).reshape(6,7), k)[1],
                    networks[int(players[j]-1)]))
    
    # winning move gets a reward of 100
    tempTargetQ[int(i-1)][int(cols[i-1])] = 100
    
    # all moves associated with losing get a reward of -100
    for j in range(outputSize):
        if j != cols[int(i-1)]:
            tempTargetQ[int(i-2)][j] = -100
        else:
            
            # if agent blocks a winning move: reward of 10
            tempTargetQ[int(i-2)][j] = 10
    
    # individual agent memory
    memoryA = []
    memoryB = []
    
    # appending value to these memory lists
    for j in range(i):
        if players[j] == 1:
            memoryA.append([boards[j],tempTargetQ[j]])
        if players[j] == 2:
            memoryB.append([boards[j],tempTargetQ[j]])
    
    # duplicating winning/losing moves in memory
    memoryA.append(memoryA[countA-1])
    memoryB.append(memoryB[countB-1])
    
    # concatenation of two memories
    gameMemory = [memoryA,memoryB]

    return gameMemory
```

# Neural Net Logic
The following three functions directly access our neural network. The first two functions work in conjunction in order to calculate and produce valid outputs for our agents. The third function deals with sampling memory in a random distribution and training on this sampled data.
<br><br>
The first function, getFull, is a "getter" function for those familiar with object oriented programming. This function is responsible for looking at the current board and determining which columns are valid moves. If a column is already full, it is invalid.
<br><br>
The second function, throughNetwork, allows access to our neural network in order to "predict" or produce a column output. This function takes the board and the network as parameters and reshapes our board to be input for the Keras predict function. We then call our previously mentioned getFull function and fix the values it returns to be the new min of network output (because those moves are invalid). This output returned is the q-values associated with each move (columns 0-6).
<br><br>
The Third function, trainNetwork, is responsible for sampling data from our memory and training on these state-target pairs. To get this sample, we take a portion of the game memories and a small subset of state-target pairs from each game. Once we get these samples, we can train on the pairs.


```python
def getFull(board):
    
    # list of full/invalid columns
    full = []
    
    for i in range(outputSize):
        if np.sum(board.reshape(6,7).T[i] != 0) == 6:
            full.append(i)
    
    return full
```


```python
def throughNetwork(board, network):
    
    # reshaping board for network input
    inputBoard = board.reshape(inputSize)
    inputBoard = board.reshape(1,-1)
    
    # getting network prediction
    output = network.predict(inputBoard)[0]
    output[getFull(inputBoard)] = min(output) - 0.1
    
    return output
```


```python
def trainNetwork(networks, memory):
    
    # size of memory sample
    sampleSize = len(memory)//10
    memPerGamePerAgent = 2
    
    # creating sample memory arrays
    sampleStateMemoryA = np.zeros((memPerGamePerAgent * sampleSize,inputSize))
    sampleStateMemoryB = np.zeros((memPerGamePerAgent * sampleSize,inputSize))
    sampleTargetA = np.zeros((memPerGamePerAgent * sampleSize,outputSize))
    sampleTargetB = np.zeros((memPerGamePerAgent * sampleSize,outputSize))
    randomGames = np.array(random.sample(range(0,len(memory)),sampleSize))

    # memory stores games
    # memory[0] is the 0th game
    # memory[0][0] is boards and targets for agent A game 0
    # memory[0][0][0] is board and target for agent A game 0, move 0
    # memory[0][0][0][0] is board for agent A game 0, move 0
    
    for i in range(len(randomGames)):
        game = memory[randomGames[i]]
        randomMovesA = np.array(random.sample(range(0,len(game[0])),memPerGamePerAgent))
        randomMovesB = np.array(random.sample(range(0,len(game[1])),memPerGamePerAgent))
        
        for j in range(memPerGamePerAgent):
            k = (i * memPerGamePerAgent) + j
            sampleStateMemoryA[k] = game[0][randomMovesA[j]][0] # This is Agent A
            sampleStateMemoryB[k] = game[1][randomMovesB[j]][0] # This is Agent B
            sampleTargetA[k] = game[0][randomMovesA[j]][1] # This is Agent A
            sampleTargetB[k] = game[1][randomMovesB[j]][1] # This is Agent B
            
    # training network
    print("Player 1 Agent...")
    networks[0].fit(sampleStateMemoryA,sampleTargetA,batch_size=16,epochs=epochs)
    print("Player 2 Agent...")
    networks[1].fit(sampleStateMemoryB,sampleTargetB,batch_size=16,epochs=epochs)
```

# Main Function
Responsible for imports of packages/frameworks, parameter definitions, and training loop.


```python
import pandas as pd
import numpy as np
import random
import time

from keras import models
from keras import layers
from keras.layers import Dense, Activation, Dropout
from keras.models import model_from_json
```

    Using TensorFlow backend.
    


```python
inputSize = 42
outputSize = 7
memSize = 5000
batchSize = 2

# memory stored by games
memory = []

memoryCount = 0
```


```python
networks = initializeNetworks(batchSize)
```

    WARNING: Logging before flag parsing goes to stderr.
    W1224 23:50:48.016955 14592 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W1224 23:50:48.030919 14592 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W1224 23:50:48.032914 14592 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    W1224 23:50:48.075803 14592 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    


```python
# epsilon definitions
epsilon = 0.5
minEpsilon = 0.1
epsilonDecay = 0.001

# iterations determind by epsilon decay rate
iterations = int(epsilon // epsilonDecay) 

# parameters for epochs and games per iteration
epochs = 4
games = 100

# timer
startTime = time.time()

# loop through training
for iteration in range(iterations):
    iterationStartTime = time.time()
    print("\nITERATION:", iteration + 1,"out of",iterations)
    
    # create gameplay, epsilon decay, and training
    memoryCount,memory = createGameplay(networks,games,memoryCount,memory,epsilon)
    if epsilon > minEpsilon:
        epsilon-=epsilonDecay
    trainNetwork(networks,memory)
    
    # stats display every iteration
    printStats(1,memoryCount,epsilon,iterationStartTime,iteration,iterations)
```

    
    ITERATION: 1 out of 50
    Player 1 Agent...
    Epoch 1/4
    4/4 [==============================] - 0s 498us/step - loss: 256.0627 - acc: 0.5000
    Epoch 2/4
    4/4 [==============================] - 0s 499us/step - loss: 245.6026 - acc: 0.5000
    Epoch 3/4
    4/4 [==============================] - 0s 497us/step - loss: 233.2798 - acc: 0.5000
    Epoch 4/4
    4/4 [==============================] - 0s 499us/step - loss: 219.9707 - acc: 0.5000
    Player 2 Agent...
    Epoch 1/4
    4/4 [==============================] - 0s 498us/step - loss: 1743.9592 - acc: 0.5000
    Epoch 2/4
    4/4 [==============================] - 0s 498us/step - loss: 1731.4989 - acc: 0.5000
    Epoch 3/4
    4/4 [==============================] - 0s 499us/step - loss: 1703.5359 - acc: 0.5000
    Epoch 4/4
    4/4 [==============================] - 0s 498us/step - loss: 1664.5153 - acc: 0.5000
    
    Games Elapsed: 199
    Current Epsilon: 49.90 percent
    
    Total Training duration: 0.17 minutes
    Current Iteration duration: 10.35 seconds
    Estimated Time remaining: 8.62 minutes
    
    ITERATION: 2 out of 50
    Player 1 Agent...
    Epoch 1/4
    6/6 [==============================] - 0s 333us/step - loss: 204.4194 - acc: 0.1667
    Epoch 2/4
    6/6 [==============================] - 0s 161us/step - loss: 200.1882 - acc: 0.1667
    Epoch 3/4
    6/6 [==============================] - 0s 326us/step - loss: 194.8353 - acc: 0.1667
    Epoch 4/4
    6/6 [==============================] - 0s 326us/step - loss: 188.6447 - acc: 0.1667
    Player 2 Agent...
    Epoch 1/4
    6/6 [==============================] - 0s 327us/step - loss: 413.2660 - acc: 0.3333
    Epoch 2/4
    6/6 [==============================] - 0s 331us/step - loss: 404.4478 - acc: 0.3333
    Epoch 3/4
    6/6 [==============================] - 0s 333us/step - loss: 393.5914 - acc: 0.3333
    Epoch 4/4
    6/6 [==============================] - 0s 332us/step - loss: 381.2713 - acc: 0.3333
    
    Games Elapsed: 298
    Current Epsilon: 49.80 percent
    
    Total Training duration: 0.31 minutes
    Current Iteration duration: 8.41 seconds
    Estimated Time remaining: 6.87 minutes
    
    ITERATION: 3 out of 50
    Player 1 Agent...
    Epoch 1/4
    8/8 [==============================] - 0s 250us/step - loss: 1323.2795 - acc: 0.1250
    Epoch 2/4
    8/8 [==============================] - 0s 125us/step - loss: 1323.1418 - acc: 0.1250
    Epoch 3/4
    8/8 [==============================] - 0s 249us/step - loss: 1318.4574 - acc: 0.1250
    Epoch 4/4
    8/8 [==============================] - 0s 125us/step - loss: 1310.1515 - acc: 0.1250
    Player 2 Agent...
    Epoch 1/4
    8/8 [==============================] - 0s 249us/step - loss: 1557.1824 - acc: 0.3750
    Epoch 2/4
    8/8 [==============================] - 0s 249us/step - loss: 1548.6544 - acc: 0.3750
    Epoch 3/4
    8/8 [==============================] - 0s 249us/step - loss: 1538.6488 - acc: 0.3750
    Epoch 4/4
    8/8 [==============================] - 0s 250us/step - loss: 1527.3196 - acc: 0.3750
    
    Games Elapsed: 398
    Current Epsilon: 49.70 percent
    
    Total Training duration: 0.45 minutes
    Current Iteration duration: 8.44 seconds
    Estimated Time remaining: 6.75 minutes
    
    ITERATION: 4 out of 50
    Player 1 Agent...
    Epoch 1/4
    10/10 [==============================] - 0s 195us/step - loss: 1269.6938 - acc: 0.5000
    Epoch 2/4
    10/10 [==============================] - 0s 100us/step - loss: 1260.9033 - acc: 0.5000
    Epoch 3/4
    10/10 [==============================] - 0s 199us/step - loss: 1247.0837 - acc: 0.5000
    Epoch 4/4
    10/10 [==============================] - 0s 299us/step - loss: 1229.5813 - acc: 0.5000
    Player 2 Agent...
    Epoch 1/4
    10/10 [==============================] - 0s 200us/step - loss: 1150.7090 - acc: 0.5000
    Epoch 2/4
    10/10 [==============================] - 0s 100us/step - loss: 1149.6351 - acc: 0.5000
    Epoch 3/4
    10/10 [==============================] - 0s 200us/step - loss: 1140.3259 - acc: 0.5000
    Epoch 4/4
    10/10 [==============================] - 0s 399us/step - loss: 1124.5911 - acc: 0.5000
    
    Games Elapsed: 498
    Current Epsilon: 49.60 percent
    
    Total Training duration: 0.60 minutes
    Current Iteration duration: 8.61 seconds
    Estimated Time remaining: 6.74 minutes
    


```python
# we can visualize the gameplay of a fully trained agent
testAgent(1,networks)
```

    Agent Move:
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 1]]
    Column(0-6):1
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 2 0 0 0 0 1]]
    Agent Move:
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 1]
     [0 2 0 0 0 0 1]]
    Column(0-6):1
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 2 0 0 0 0 1]
     [0 2 0 0 0 0 1]]
    Agent Move:
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0]
     [0 2 0 0 0 0 1]
     [0 2 0 0 0 0 1]]
    Column(0-6):2
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0]
     [0 2 0 0 0 0 1]
     [0 2 2 0 0 0 1]]
    Agent Move:
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0]
     [0 2 0 0 0 0 1]
     [0 2 2 0 0 1 1]]
    Column(0-6):2
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0]
     [0 2 2 0 0 0 1]
     [0 2 2 0 0 1 1]]
    Agent Move:
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0]
     [0 2 2 0 0 0 1]
     [0 2 2 0 1 1 1]]
    Column(0-6):2
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 1 2 0 0 0 0]
     [0 2 2 0 0 0 1]
     [0 2 2 0 1 1 1]]
    Agent Move:
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0]
     [0 1 2 0 0 0 0]
     [0 2 2 0 0 0 1]
     [0 2 2 0 1 1 1]]
    Column(0-6):3
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0]
     [0 1 2 0 0 0 0]
     [0 2 2 0 0 0 1]
     [0 2 2 2 1 1 1]]
    Agent Move:
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 1 1 0 0 0 0]
     [0 1 2 0 0 0 0]
     [0 2 2 0 0 0 1]
     [0 2 2 2 1 1 1]]
    Column(0-6):3
    [[0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0]
     [0 1 1 0 0 0 0]
     [0 1 2 0 0 0 0]
     [0 2 2 2 0 0 1]
     [0 2 2 2 1 1 1]]
    Agent Move:
    [[0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0]
     [0 1 1 0 0 0 0]
     [0 1 2 0 0 0 0]
     [0 2 2 2 0 0 1]
     [0 2 2 2 1 1 1]]
    Column(0-6):3
    [[0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0]
     [0 1 1 0 0 0 0]
     [0 1 2 2 0 0 0]
     [0 2 2 2 0 0 1]
     [0 2 2 2 1 1 1]]
    Agent Move:
    [[0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0]
     [0 1 1 1 0 0 0]
     [0 1 2 2 0 0 0]
     [0 2 2 2 0 0 1]
     [0 2 2 2 1 1 1]]
    Column(0-6):4
    [[0 0 0 0 0 0 0]
     [0 0 1 0 0 0 0]
     [0 1 1 1 0 0 0]
     [0 1 2 2 0 0 0]
     [0 2 2 2 2 0 1]
     [0 2 2 2 1 1 1]]
    


```python
def printStats(interval,memoryCount,epsilon,iterationStartTime,iteration,iterations):
    
    # statustics for training loop
    if iteration % interval == 0:
        print("\nGames Elapsed:",memoryCount)
        print("Current Epsilon:",format(epsilon*100,'.2f'),"percent")
        print("\nTotal Training duration:", format((time.time() - startTime)/60,'.2f'),"minutes")
        print("Current Iteration duration:", format((time.time() - iterationStartTime),'.2f'),"seconds")
        print("Estimated Time remaining:",format((time.time() - iterationStartTime)*(iterations-iteration)/60,'.2f'),"minutes")
```


```python
def testAgent(agent,networks):
    board = createBoard()
    win = False
    tieGame = False
    while win == False:
        for player in range(1,3):
            if player == agent:
                try:
                    col = epsilonGreedy(0.0,networks[player-1],board)
                    win,board,col,i = step(player,board,col)
                    print("Agent Move:")
                    print(board)
                    # tie
                    if col == -1 or i == -7:
                        tieGame = True
                        break
                    
                    if win == True:
                        break
                        
                except ValueError:
                    pass
            else:
                col = int(input("Column(0-6):"))
                win,board,col,i = step(player,board,col)
                print(board)
                
                if win == True:
                    break
```

# Save/Load weights


```python
# Save

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

    Saved model 1 to disk
    Saved model 2 to disk
    


```python
# Load

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

# copile network post-load
networks[0].compile(optimizer='adam',loss='mse',metrics=['accuracy'])
networks[1].compile(optimizer='adam',loss='mse',metrics=['accuracy'])
```

    W1224 23:50:53.420687 14592 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:174: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
    
    W1224 23:50:53.421686 14592 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:181: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    

    Loaded Model from disk
    Loaded Model from disk
    
