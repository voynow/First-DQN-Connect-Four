
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
    
    ITERATION: 5 out of 50
    Player 1 Agent...
    Epoch 1/4
    12/12 [==============================] - 0s 252us/step - loss: 2230.5137 - acc: 0.3333
    Epoch 2/4
    12/12 [==============================] - 0s 164us/step - loss: 2219.3210 - acc: 0.3333
    Epoch 3/4
    12/12 [==============================] - 0s 83us/step - loss: 2202.4441 - acc: 0.3333
    Epoch 4/4
    12/12 [==============================] - 0s 166us/step - loss: 2181.3513 - acc: 0.3333
    Player 2 Agent...
    Epoch 1/4
    12/12 [==============================] - 0s 166us/step - loss: 634.6804 - acc: 0.1667
    Epoch 2/4
    12/12 [==============================] - 0s 83us/step - loss: 621.9382 - acc: 0.1667
    Epoch 3/4
    12/12 [==============================] - 0s 167us/step - loss: 606.6545 - acc: 0.1667
    Epoch 4/4
    12/12 [==============================] - 0s 81us/step - loss: 589.6915 - acc: 0.1667
    
    Games Elapsed: 597
    Current Epsilon: 49.50 percent
    
    Total Training duration: 0.73 minutes
    Current Iteration duration: 7.92 seconds
    Estimated Time remaining: 6.07 minutes
    
    ITERATION: 6 out of 50
    Player 1 Agent...
    Epoch 1/4
    14/14 [==============================] - 0s 69us/step - loss: 1040.9518 - acc: 0.4286
    Epoch 2/4
    14/14 [==============================] - 0s 142us/step - loss: 1032.3124 - acc: 0.4286
    Epoch 3/4
    14/14 [==============================] - 0s 142us/step - loss: 1021.4245 - acc: 0.4286
    Epoch 4/4
    14/14 [==============================] - 0s 143us/step - loss: 1008.9202 - acc: 0.4286
    Player 2 Agent...
    Epoch 1/4
    14/14 [==============================] - 0s 139us/step - loss: 1701.1888 - acc: 0.2143
    Epoch 2/4
    14/14 [==============================] - 0s 214us/step - loss: 1696.6740 - acc: 0.2143
    Epoch 3/4
    14/14 [==============================] - 0s 143us/step - loss: 1690.3419 - acc: 0.1429
    Epoch 4/4
    14/14 [==============================] - 0s 143us/step - loss: 1682.3883 - acc: 0.1429
    
    Games Elapsed: 697
    Current Epsilon: 49.40 percent
    
    Total Training duration: 0.84 minutes
    Current Iteration duration: 6.90 seconds
    Estimated Time remaining: 5.18 minutes
    
    ITERATION: 7 out of 50
    Player 1 Agent...
    Epoch 1/4
    16/16 [==============================] - 0s 121us/step - loss: 1845.6152 - acc: 0.4375
    Epoch 2/4
    16/16 [==============================] - 0s 62us/step - loss: 1834.7776 - acc: 0.4375
    Epoch 3/4
    16/16 [==============================] - 0s 125us/step - loss: 1821.2312 - acc: 0.4375
    Epoch 4/4
    16/16 [==============================] - 0s 62us/step - loss: 1805.4373 - acc: 0.4375
    Player 2 Agent...
    Epoch 1/4
    16/16 [==============================] - 0s 125us/step - loss: 1053.4176 - acc: 0.0625
    Epoch 2/4
    16/16 [==============================] - 0s 127us/step - loss: 1046.0623 - acc: 0.0625
    Epoch 3/4
    16/16 [==============================] - 0s 60us/step - loss: 1036.6487 - acc: 0.0625
    Epoch 4/4
    16/16 [==============================] - 0s 125us/step - loss: 1025.5098 - acc: 0.0625
    
    Games Elapsed: 797
    Current Epsilon: 49.30 percent
    
    Total Training duration: 0.97 minutes
    Current Iteration duration: 7.61 seconds
    Estimated Time remaining: 5.58 minutes
    
    ITERATION: 8 out of 50
    Player 1 Agent...
    Epoch 1/4
    18/18 [==============================] - 0s 166us/step - loss: 757.7891 - acc: 0.3333
    Epoch 2/4
    18/18 [==============================] - 0s 222us/step - loss: 755.9041 - acc: 0.3333
    Epoch 3/4
    18/18 [==============================] - 0s 222us/step - loss: 751.9195 - acc: 0.3333
    Epoch 4/4
    18/18 [==============================] - 0s 222us/step - loss: 745.2466 - acc: 0.2778
    Player 2 Agent...
    Epoch 1/4
    18/18 [==============================] - 0s 277us/step - loss: 2210.3658 - acc: 0.2778
    Epoch 2/4
    18/18 [==============================] - 0s 665us/step - loss: 2186.3277 - acc: 0.2778
    Epoch 3/4
    18/18 [==============================] - 0s 222us/step - loss: 2142.3054 - acc: 0.2778
    Epoch 4/4
    18/18 [==============================] - 0s 220us/step - loss: 2099.4516 - acc: 0.2778
    
    Games Elapsed: 896
    Current Epsilon: 49.20 percent
    
    Total Training duration: 1.09 minutes
    Current Iteration duration: 7.08 seconds
    Estimated Time remaining: 5.08 minutes
    
    ITERATION: 9 out of 50
    Player 1 Agent...
    Epoch 1/4
    20/20 [==============================] - 0s 247us/step - loss: 681.9450 - acc: 0.1500
    Epoch 2/4
    20/20 [==============================] - 0s 198us/step - loss: 684.7306 - acc: 0.1500
    Epoch 3/4
    20/20 [==============================] - 0s 200us/step - loss: 680.0320 - acc: 0.1500
    Epoch 4/4
    20/20 [==============================] - 0s 199us/step - loss: 669.6771 - acc: 0.1500
    Player 2 Agent...
    Epoch 1/4
    20/20 [==============================] - 0s 349us/step - loss: 1132.5875 - acc: 0.1000
    Epoch 2/4
    20/20 [==============================] - 0s 448us/step - loss: 1125.1360 - acc: 0.1000
    Epoch 3/4
    20/20 [==============================] - 0s 299us/step - loss: 1113.7739 - acc: 0.1000
    Epoch 4/4
    20/20 [==============================] - 0s 249us/step - loss: 1100.7344 - acc: 0.1000
    
    Games Elapsed: 996
    Current Epsilon: 49.10 percent
    
    Total Training duration: 1.23 minutes
    Current Iteration duration: 8.65 seconds
    Estimated Time remaining: 6.05 minutes
    
    ITERATION: 10 out of 50
    Player 1 Agent...
    Epoch 1/4
    22/22 [==============================] - 0s 136us/step - loss: 1480.8227 - acc: 0.4545
    Epoch 2/4
    22/22 [==============================] - 0s 181us/step - loss: 1470.8326 - acc: 0.5000
    Epoch 3/4
    22/22 [==============================] - 0s 227us/step - loss: 1456.7199 - acc: 0.5000
    Epoch 4/4
    22/22 [==============================] - 0s 136us/step - loss: 1423.8426 - acc: 0.5000
    Player 2 Agent...
    Epoch 1/4
    22/22 [==============================] - 0s 453us/step - loss: 1365.7267 - acc: 0.4091
    Epoch 2/4
    22/22 [==============================] - 0s 227us/step - loss: 1346.8290 - acc: 0.4091
    Epoch 3/4
    22/22 [==============================] - 0s 453us/step - loss: 1326.7400 - acc: 0.4091
    Epoch 4/4
    22/22 [==============================] - 0s 182us/step - loss: 1301.0839 - acc: 0.4091
    
    Games Elapsed: 1095
    Current Epsilon: 49.00 percent
    
    Total Training duration: 1.36 minutes
    Current Iteration duration: 7.48 seconds
    Estimated Time remaining: 5.12 minutes
    
    ITERATION: 11 out of 50
    Player 1 Agent...
    Epoch 1/4
    24/24 [==============================] - 0s 125us/step - loss: 896.5085 - acc: 0.3333
    Epoch 2/4
    24/24 [==============================] - 0s 166us/step - loss: 886.8480 - acc: 0.3333
    Epoch 3/4
    24/24 [==============================] - 0s 208us/step - loss: 872.8590 - acc: 0.3333
    Epoch 4/4
    24/24 [==============================] - 0s 208us/step - loss: 854.6390 - acc: 0.3333
    Player 2 Agent...
    Epoch 1/4
    24/24 [==============================] - 0s 208us/step - loss: 978.4882 - acc: 0.2917
    Epoch 2/4
    24/24 [==============================] - 0s 166us/step - loss: 978.6751 - acc: 0.2917
    Epoch 3/4
    24/24 [==============================] - 0s 208us/step - loss: 963.7576 - acc: 0.2917
    Epoch 4/4
    24/24 [==============================] - 0s 208us/step - loss: 937.3068 - acc: 0.2917
    
    Games Elapsed: 1195
    Current Epsilon: 48.90 percent
    
    Total Training duration: 1.49 minutes
    Current Iteration duration: 7.94 seconds
    Estimated Time remaining: 5.29 minutes
    
    ITERATION: 12 out of 50
    Player 1 Agent...
    Epoch 1/4
    26/26 [==============================] - 0s 192us/step - loss: 619.2991 - acc: 0.3462
    Epoch 2/4
    26/26 [==============================] - 0s 153us/step - loss: 617.7695 - acc: 0.3462
    Epoch 3/4
    26/26 [==============================] - 0s 192us/step - loss: 613.0828 - acc: 0.3462
    Epoch 4/4
    26/26 [==============================] - 0s 267us/step - loss: 607.8795 - acc: 0.3462
    Player 2 Agent...
    Epoch 1/4
    26/26 [==============================] - 0s 228us/step - loss: 1886.6986 - acc: 0.1923
    Epoch 2/4
    26/26 [==============================] - 0s 152us/step - loss: 1874.8384 - acc: 0.2308
    Epoch 3/4
    26/26 [==============================] - 0s 192us/step - loss: 1849.3229 - acc: 0.2692
    Epoch 4/4
    26/26 [==============================] - 0s 192us/step - loss: 1823.9122 - acc: 0.3077
    
    Games Elapsed: 1295
    Current Epsilon: 48.80 percent
    
    Total Training duration: 1.60 minutes
    Current Iteration duration: 6.82 seconds
    Estimated Time remaining: 4.44 minutes
    
    ITERATION: 13 out of 50
    Player 1 Agent...
    Epoch 1/4
    28/28 [==============================] - 0s 143us/step - loss: 1847.6470 - acc: 0.2857
    Epoch 2/4
    28/28 [==============================] - 0s 143us/step - loss: 1838.3161 - acc: 0.2857
    Epoch 3/4
    28/28 [==============================] - 0s 142us/step - loss: 1817.2613 - acc: 0.3214
    Epoch 4/4
    28/28 [==============================] - 0s 179us/step - loss: 1786.1935 - acc: 0.3214
    Player 2 Agent...
    Epoch 1/4
    28/28 [==============================] - 0s 321us/step - loss: 1384.5352 - acc: 0.2500
    Epoch 2/4
    28/28 [==============================] - 0s 143us/step - loss: 1383.2784 - acc: 0.2500
    Epoch 3/4
    28/28 [==============================] - 0s 212us/step - loss: 1378.4741 - acc: 0.2500
    Epoch 4/4
    28/28 [==============================] - 0s 285us/step - loss: 1366.0151 - acc: 0.2500
    
    Games Elapsed: 1395
    Current Epsilon: 48.70 percent
    
    Total Training duration: 1.74 minutes
    Current Iteration duration: 8.28 seconds
    Estimated Time remaining: 5.25 minutes
    
    ITERATION: 14 out of 50
    Player 1 Agent...
    Epoch 1/4
    30/30 [==============================] - 0s 133us/step - loss: 523.9960 - acc: 0.3333
    Epoch 2/4
    30/30 [==============================] - 0s 166us/step - loss: 522.2775 - acc: 0.3333
    Epoch 3/4
    30/30 [==============================] - 0s 133us/step - loss: 518.1653 - acc: 0.3333
    Epoch 4/4
    30/30 [==============================] - 0s 166us/step - loss: 509.3542 - acc: 0.3333
    Player 2 Agent...
    Epoch 1/4
    30/30 [==============================] - 0s 199us/step - loss: 1417.3081 - acc: 0.3000
    Epoch 2/4
    30/30 [==============================] - 0s 133us/step - loss: 1414.8888 - acc: 0.3000
    Epoch 3/4
    30/30 [==============================] - 0s 166us/step - loss: 1407.8574 - acc: 0.3000
    Epoch 4/4
    30/30 [==============================] - 0s 133us/step - loss: 1397.6448 - acc: 0.3000
    
    Games Elapsed: 1495
    Current Epsilon: 48.60 percent
    
    Total Training duration: 1.86 minutes
    Current Iteration duration: 7.13 seconds
    Estimated Time remaining: 4.40 minutes
    
    ITERATION: 15 out of 50
    Player 1 Agent...
    Epoch 1/4
    32/32 [==============================] - 0s 123us/step - loss: 1440.0208 - acc: 0.3750
    Epoch 2/4
    32/32 [==============================] - 0s 123us/step - loss: 1436.1166 - acc: 0.3750
    Epoch 3/4
    32/32 [==============================] - 0s 125us/step - loss: 1426.6190 - acc: 0.3750
    Epoch 4/4
    32/32 [==============================] - 0s 125us/step - loss: 1411.5604 - acc: 0.3750
    Player 2 Agent...
    Epoch 1/4
    32/32 [==============================] - 0s 125us/step - loss: 1000.3711 - acc: 0.2500
    Epoch 2/4
    32/32 [==============================] - 0s 124us/step - loss: 997.1423 - acc: 0.2500
    Epoch 3/4
    32/32 [==============================] - 0s 249us/step - loss: 986.6742 - acc: 0.2500
    Epoch 4/4
    32/32 [==============================] - 0s 123us/step - loss: 975.2243 - acc: 0.2500
    
    Games Elapsed: 1594
    Current Epsilon: 48.50 percent
    
    Total Training duration: 2.00 minutes
    Current Iteration duration: 8.40 seconds
    Estimated Time remaining: 5.04 minutes
    
    ITERATION: 16 out of 50
    Player 1 Agent...
    Epoch 1/4
    34/34 [==============================] - 0s 116us/step - loss: 746.4842 - acc: 0.4706
    Epoch 2/4
    34/34 [==============================] - 0s 174us/step - loss: 733.9813 - acc: 0.4706
    Epoch 3/4
    34/34 [==============================] - 0s 176us/step - loss: 710.5103 - acc: 0.5000
    Epoch 4/4
    34/34 [==============================] - 0s 176us/step - loss: 678.7070 - acc: 0.5000
    Player 2 Agent...
    Epoch 1/4
    34/34 [==============================] - 0s 323us/step - loss: 1952.5932 - acc: 0.2059
    Epoch 2/4
    34/34 [==============================] - 0s 175us/step - loss: 1928.4981 - acc: 0.2353
    Epoch 3/4
    34/34 [==============================] - 0s 293us/step - loss: 1902.9724 - acc: 0.2353
    Epoch 4/4
    34/34 [==============================] - 0s 176us/step - loss: 1883.5032 - acc: 0.2353
    
    Games Elapsed: 1694
    Current Epsilon: 48.40 percent
    
    Total Training duration: 2.15 minutes
    Current Iteration duration: 9.02 seconds
    Estimated Time remaining: 5.26 minutes
    
    ITERATION: 17 out of 50
    Player 1 Agent...
    Epoch 1/4
    36/36 [==============================] - 0s 194us/step - loss: 1818.8388 - acc: 0.3056
    Epoch 2/4
    36/36 [==============================] - 0s 137us/step - loss: 1789.8599 - acc: 0.3056
    Epoch 3/4
    36/36 [==============================] - 0s 138us/step - loss: 1756.9018 - acc: 0.3056
    Epoch 4/4
    36/36 [==============================] - 0s 166us/step - loss: 1712.6535 - acc: 0.3056
    Player 2 Agent...
    Epoch 1/4
    36/36 [==============================] - 0s 165us/step - loss: 2279.3111 - acc: 0.4444
    Epoch 2/4
    36/36 [==============================] - 0s 249us/step - loss: 2228.1766 - acc: 0.4444
    Epoch 3/4
    36/36 [==============================] - 0s 138us/step - loss: 2156.9193 - acc: 0.4722
    Epoch 4/4
    36/36 [==============================] - 0s 249us/step - loss: 2079.7967 - acc: 0.5000
    
    Games Elapsed: 1794
    Current Epsilon: 48.30 percent
    
    Total Training duration: 2.28 minutes
    Current Iteration duration: 7.84 seconds
    Estimated Time remaining: 4.44 minutes
    
    ITERATION: 18 out of 50
    Player 1 Agent...
    Epoch 1/4
    38/38 [==============================] - 0s 167us/step - loss: 1818.5169 - acc: 0.1579
    Epoch 2/4
    38/38 [==============================] - 0s 157us/step - loss: 1797.9600 - acc: 0.1842
    Epoch 3/4
    38/38 [==============================] - 0s 158us/step - loss: 1751.0875 - acc: 0.1842
    Epoch 4/4
    38/38 [==============================] - 0s 291us/step - loss: 1656.7475 - acc: 0.2368
    Player 2 Agent...
    Epoch 1/4
    38/38 [==============================] - 0s 182us/step - loss: 1190.7729 - acc: 0.3158
    Epoch 2/4
    38/38 [==============================] - 0s 289us/step - loss: 1225.7815 - acc: 0.3158
    Epoch 3/4
    38/38 [==============================] - 0s 184us/step - loss: 1138.5323 - acc: 0.2895
    Epoch 4/4
    38/38 [==============================] - 0s 262us/step - loss: 986.0847 - acc: 0.2895
    
    Games Elapsed: 1894
    Current Epsilon: 48.20 percent
    
    Total Training duration: 2.41 minutes
    Current Iteration duration: 7.71 seconds
    Estimated Time remaining: 4.24 minutes
    
    ITERATION: 19 out of 50
    Player 1 Agent...
    Epoch 1/4
    40/40 [==============================] - 0s 149us/step - loss: 941.1258 - acc: 0.2500
    Epoch 2/4
    40/40 [==============================] - 0s 127us/step - loss: 944.7197 - acc: 0.2500
    Epoch 3/4
    40/40 [==============================] - 0s 125us/step - loss: 932.7788 - acc: 0.2750
    Epoch 4/4
    40/40 [==============================] - 0s 150us/step - loss: 921.5190 - acc: 0.3250
    Player 2 Agent...
    Epoch 1/4
    40/40 [==============================] - 0s 249us/step - loss: 670.9620 - acc: 0.2750
    Epoch 2/4
    40/40 [==============================] - 0s 124us/step - loss: 645.8901 - acc: 0.2750
    Epoch 3/4
    40/40 [==============================] - 0s 174us/step - loss: 622.9745 - acc: 0.2750
    Epoch 4/4
    40/40 [==============================] - 0s 175us/step - loss: 609.1672 - acc: 0.2750
    
    Games Elapsed: 1994
    Current Epsilon: 48.10 percent
    
    Total Training duration: 2.57 minutes
    Current Iteration duration: 9.40 seconds
    Estimated Time remaining: 5.01 minutes
    
    ITERATION: 20 out of 50
    Player 1 Agent...
    Epoch 1/4
    42/42 [==============================] - 0s 126us/step - loss: 746.6958 - acc: 0.3095
    Epoch 2/4
    42/42 [==============================] - 0s 142us/step - loss: 711.2084 - acc: 0.3095
    Epoch 3/4
    42/42 [==============================] - 0s 142us/step - loss: 668.7969 - acc: 0.3095
    Epoch 4/4
    42/42 [==============================] - 0s 261us/step - loss: 632.7907 - acc: 0.4286
    Player 2 Agent...
    Epoch 1/4
    42/42 [==============================] - 0s 142us/step - loss: 1670.1111 - acc: 0.3571
    Epoch 2/4
    42/42 [==============================] - 0s 190us/step - loss: 1656.4430 - acc: 0.3571
    Epoch 3/4
    42/42 [==============================] - 0s 166us/step - loss: 1637.0225 - acc: 0.3571
    Epoch 4/4
    42/42 [==============================] - 0s 261us/step - loss: 1596.7850 - acc: 0.3571
    
    Games Elapsed: 2094
    Current Epsilon: 48.00 percent
    
    Total Training duration: 2.71 minutes
    Current Iteration duration: 8.80 seconds
    Estimated Time remaining: 4.55 minutes
    
    ITERATION: 21 out of 50
    Player 1 Agent...
    Epoch 1/4
    44/44 [==============================] - 0s 113us/step - loss: 573.8486 - acc: 0.2273
    Epoch 2/4
    44/44 [==============================] - 0s 136us/step - loss: 550.8915 - acc: 0.2273
    Epoch 3/4
    44/44 [==============================] - 0s 113us/step - loss: 521.7132 - acc: 0.2273
    Epoch 4/4
    44/44 [==============================] - 0s 271us/step - loss: 496.4598 - acc: 0.2500
    Player 2 Agent...
    Epoch 1/4
    44/44 [==============================] - 0s 135us/step - loss: 1179.5515 - acc: 0.2500
    Epoch 2/4
    44/44 [==============================] - 0s 228us/step - loss: 1172.9899 - acc: 0.2500
    Epoch 3/4
    44/44 [==============================] - 0s 113us/step - loss: 1154.5059 - acc: 0.2955
    Epoch 4/4
    44/44 [==============================] - 0s 272us/step - loss: 1126.7948 - acc: 0.2955
    
    Games Elapsed: 2194
    Current Epsilon: 47.90 percent
    
    Total Training duration: 2.84 minutes
    Current Iteration duration: 7.72 seconds
    Estimated Time remaining: 3.86 minutes
    
    ITERATION: 22 out of 50
    Player 1 Agent...
    Epoch 1/4
    46/46 [==============================] - 0s 130us/step - loss: 528.4584 - acc: 0.3043
    Epoch 2/4
    46/46 [==============================] - 0s 130us/step - loss: 523.1653 - acc: 0.3043
    Epoch 3/4
    46/46 [==============================] - 0s 260us/step - loss: 516.8290 - acc: 0.3043
    Epoch 4/4
    46/46 [==============================] - 0s 108us/step - loss: 506.7618 - acc: 0.3043
    Player 2 Agent...
    Epoch 1/4
    46/46 [==============================] - 0s 260us/step - loss: 756.4440 - acc: 0.3478
    Epoch 2/4
    46/46 [==============================] - 0s 108us/step - loss: 752.8678 - acc: 0.3696
    Epoch 3/4
    46/46 [==============================] - 0s 217us/step - loss: 735.1146 - acc: 0.3696
    Epoch 4/4
    46/46 [==============================] - 0s 108us/step - loss: 714.1477 - acc: 0.3696
    
    Games Elapsed: 2294
    Current Epsilon: 47.80 percent
    
    Total Training duration: 2.99 minutes
    Current Iteration duration: 8.59 seconds
    Estimated Time remaining: 4.15 minutes
    
    ITERATION: 23 out of 50
    Player 1 Agent...
    Epoch 1/4
    48/48 [==============================] - 0s 104us/step - loss: 1469.0342 - acc: 0.2917
    Epoch 2/4
    48/48 [==============================] - 0s 125us/step - loss: 1447.8231 - acc: 0.2917
    Epoch 3/4
    48/48 [==============================] - 0s 104us/step - loss: 1394.7832 - acc: 0.2917
    Epoch 4/4
    48/48 [==============================] - 0s 249us/step - loss: 1363.8278 - acc: 0.2500
    Player 2 Agent...
    Epoch 1/4
    48/48 [==============================] - 0s 125us/step - loss: 785.7245 - acc: 0.3542
    Epoch 2/4
    48/48 [==============================] - 0s 228us/step - loss: 778.2953 - acc: 0.3542
    Epoch 3/4
    48/48 [==============================] - 0s 104us/step - loss: 765.9478 - acc: 0.3750
    Epoch 4/4
    48/48 [==============================] - 0s 229us/step - loss: 746.0644 - acc: 0.3750
    
    Games Elapsed: 2393
    Current Epsilon: 47.70 percent
    
    Total Training duration: 3.11 minutes
    Current Iteration duration: 7.36 seconds
    Estimated Time remaining: 3.44 minutes
    
    ITERATION: 24 out of 50
    Player 1 Agent...
    Epoch 1/4
    50/50 [==============================] - 0s 119us/step - loss: 1094.5457 - acc: 0.2800
    Epoch 2/4
    50/50 [==============================] - 0s 140us/step - loss: 1063.5519 - acc: 0.2800
    Epoch 3/4
    50/50 [==============================] - 0s 220us/step - loss: 1028.3229 - acc: 0.2600
    Epoch 4/4
    50/50 [==============================] - 0s 140us/step - loss: 989.0507 - acc: 0.2600
    Player 2 Agent...
    Epoch 1/4
    50/50 [==============================] - 0s 279us/step - loss: 1522.7662 - acc: 0.3400
    Epoch 2/4
    50/50 [==============================] - 0s 160us/step - loss: 1493.2702 - acc: 0.3400
    Epoch 3/4
    50/50 [==============================] - 0s 159us/step - loss: 1449.2622 - acc: 0.3400
    Epoch 4/4
    50/50 [==============================] - 0s 220us/step - loss: 1415.6878 - acc: 0.3800
    
    Games Elapsed: 2493
    Current Epsilon: 47.60 percent
    
    Total Training duration: 3.25 minutes
    Current Iteration duration: 8.48 seconds
    Estimated Time remaining: 3.82 minutes
    
    ITERATION: 25 out of 50
    Player 1 Agent...
    Epoch 1/4
    52/52 [==============================] - 0s 134us/step - loss: 817.3892 - acc: 0.3846
    Epoch 2/4
    52/52 [==============================] - 0s 114us/step - loss: 815.9601 - acc: 0.3846
    Epoch 3/4
    52/52 [==============================] - 0s 173us/step - loss: 807.6187 - acc: 0.3846
    Epoch 4/4
    52/52 [==============================] - 0s 134us/step - loss: 797.7583 - acc: 0.3846
    Player 2 Agent...
    Epoch 1/4
    52/52 [==============================] - 0s 211us/step - loss: 1718.3838 - acc: 0.3462
    Epoch 2/4
    52/52 [==============================] - 0s 134us/step - loss: 1676.8170 - acc: 0.3462
    Epoch 3/4
    52/52 [==============================] - 0s 230us/step - loss: 1576.4283 - acc: 0.3462
    Epoch 4/4
    52/52 [==============================] - 0s 153us/step - loss: 1515.4217 - acc: 0.3654
    
    Games Elapsed: 2593
    Current Epsilon: 47.50 percent
    
    Total Training duration: 3.38 minutes
    Current Iteration duration: 7.68 seconds
    Estimated Time remaining: 3.33 minutes
    
    ITERATION: 26 out of 50
    Player 1 Agent...
    Epoch 1/4
    54/54 [==============================] - 0s 129us/step - loss: 1087.7326 - acc: 0.3704
    Epoch 2/4
    54/54 [==============================] - 0s 148us/step - loss: 1078.1471 - acc: 0.3519
    Epoch 3/4
    54/54 [==============================] - 0s 129us/step - loss: 1064.7236 - acc: 0.3333
    Epoch 4/4
    54/54 [==============================] - 0s 259us/step - loss: 1046.2841 - acc: 0.3519
    Player 2 Agent...
    Epoch 1/4
    54/54 [==============================] - 0s 148us/step - loss: 2084.8785 - acc: 0.3333
    Epoch 2/4
    54/54 [==============================] - 0s 165us/step - loss: 2073.0739 - acc: 0.2963
    Epoch 3/4
    54/54 [==============================] - 0s 240us/step - loss: 2036.6463 - acc: 0.3333
    Epoch 4/4
    54/54 [==============================] - 0s 259us/step - loss: 1979.3356 - acc: 0.3519
    
    Games Elapsed: 2693
    Current Epsilon: 47.40 percent
    
    Total Training duration: 3.52 minutes
    Current Iteration duration: 8.74 seconds
    Estimated Time remaining: 3.64 minutes
    
    ITERATION: 27 out of 50
    Player 1 Agent...
    Epoch 1/4
    56/56 [==============================] - 0s 125us/step - loss: 2150.9486 - acc: 0.3393
    Epoch 2/4
    56/56 [==============================] - 0s 125us/step - loss: 2100.0419 - acc: 0.3393
    Epoch 3/4
    56/56 [==============================] - 0s 142us/step - loss: 2016.9625 - acc: 0.3214
    Epoch 4/4
    56/56 [==============================] - 0s 214us/step - loss: 1934.2026 - acc: 0.3214
    Player 2 Agent...
    Epoch 1/4
    56/56 [==============================] - 0s 142us/step - loss: 1289.4252 - acc: 0.2857
    Epoch 2/4
    56/56 [==============================] - 0s 232us/step - loss: 1269.8484 - acc: 0.3036
    Epoch 3/4
    56/56 [==============================] - 0s 249us/step - loss: 1237.9862 - acc: 0.3214
    Epoch 4/4
    56/56 [==============================] - 0s 267us/step - loss: 1210.1327 - acc: 0.3214
    
    Games Elapsed: 2792
    Current Epsilon: 47.30 percent
    
    Total Training duration: 3.67 minutes
    Current Iteration duration: 9.00 seconds
    Estimated Time remaining: 3.60 minutes
    
    ITERATION: 28 out of 50
    Player 1 Agent...
    Epoch 1/4
    58/58 [==============================] - 0s 123us/step - loss: 1117.8794 - acc: 0.3448
    Epoch 2/4
    58/58 [==============================] - 0s 138us/step - loss: 1102.8316 - acc: 0.3448
    Epoch 3/4
    58/58 [==============================] - 0s 241us/step - loss: 1052.0605 - acc: 0.3448
    Epoch 4/4
    58/58 [==============================] - 0s 206us/step - loss: 996.6366 - acc: 0.3621
    Player 2 Agent...
    Epoch 1/4
    58/58 [==============================] - 0s 120us/step - loss: 1636.9144 - acc: 0.4310
    Epoch 2/4
    58/58 [==============================] - 0s 224us/step - loss: 1548.2025 - acc: 0.4138
    Epoch 3/4
    58/58 [==============================] - 0s 206us/step - loss: 1453.9878 - acc: 0.4138
    Epoch 4/4
    58/58 [==============================] - 0s 120us/step - loss: 1327.7738 - acc: 0.3966
    
    Games Elapsed: 2892
    Current Epsilon: 47.20 percent
    
    Total Training duration: 3.83 minutes
    Current Iteration duration: 9.29 seconds
    Estimated Time remaining: 3.56 minutes
    
    ITERATION: 29 out of 50
    Player 1 Agent...
    Epoch 1/4
    60/60 [==============================] - 0s 133us/step - loss: 695.6058 - acc: 0.2500
    Epoch 2/4
    60/60 [==============================] - 0s 149us/step - loss: 648.9058 - acc: 0.2500
    Epoch 3/4
    60/60 [==============================] - 0s 133us/step - loss: 608.4980 - acc: 0.2500
    Epoch 4/4
    60/60 [==============================] - 0s 182us/step - loss: 581.0825 - acc: 0.2333
    Player 2 Agent...
    Epoch 1/4
    60/60 [==============================] - 0s 116us/step - loss: 1140.3077 - acc: 0.3500
    Epoch 2/4
    60/60 [==============================] - 0s 150us/step - loss: 1074.7395 - acc: 0.3667
    Epoch 3/4
    60/60 [==============================] - 0s 215us/step - loss: 1014.5128 - acc: 0.3833
    Epoch 4/4
    60/60 [==============================] - 0s 116us/step - loss: 963.8445 - acc: 0.3667
    
    Games Elapsed: 2992
    Current Epsilon: 47.10 percent
    
    Total Training duration: 3.98 minutes
    Current Iteration duration: 9.38 seconds
    Estimated Time remaining: 3.44 minutes
    
    ITERATION: 30 out of 50
    Player 1 Agent...
    Epoch 1/4
    62/62 [==============================] - 0s 112us/step - loss: 807.5993 - acc: 0.2742
    Epoch 2/4
    62/62 [==============================] - 0s 97us/step - loss: 798.0052 - acc: 0.2742
    Epoch 3/4
    62/62 [==============================] - 0s 193us/step - loss: 774.5442 - acc: 0.2742
    Epoch 4/4
    62/62 [==============================] - 0s 113us/step - loss: 753.1640 - acc: 0.3065
    Player 2 Agent...
    Epoch 1/4
    62/62 [==============================] - 0s 193us/step - loss: 1149.4665 - acc: 0.3065
    Epoch 2/4
    62/62 [==============================] - 0s 113us/step - loss: 1112.6195 - acc: 0.3065
    Epoch 3/4
    62/62 [==============================] - 0s 177us/step - loss: 1080.4461 - acc: 0.3065
    Epoch 4/4
    62/62 [==============================] - 0s 177us/step - loss: 1040.7789 - acc: 0.3226
    
    Games Elapsed: 3092
    Current Epsilon: 47.00 percent
    
    Total Training duration: 4.12 minutes
    Current Iteration duration: 8.13 seconds
    Estimated Time remaining: 2.84 minutes
    
    ITERATION: 31 out of 50
    Player 1 Agent...
    Epoch 1/4
    64/64 [==============================] - 0s 125us/step - loss: 1301.6266 - acc: 0.3438
    Epoch 2/4
    64/64 [==============================] - 0s 108us/step - loss: 1280.4841 - acc: 0.3594
    Epoch 3/4
    64/64 [==============================] - 0s 172us/step - loss: 1240.8241 - acc: 0.3750
    Epoch 4/4
    64/64 [==============================] - 0s 109us/step - loss: 1205.8803 - acc: 0.3750
    Player 2 Agent...
    Epoch 1/4
    64/64 [==============================] - 0s 171us/step - loss: 1186.2912 - acc: 0.3281
    Epoch 2/4
    64/64 [==============================] - 0s 109us/step - loss: 1175.5920 - acc: 0.3281
    Epoch 3/4
    64/64 [==============================] - 0s 203us/step - loss: 1156.4240 - acc: 0.3281
    Epoch 4/4
    64/64 [==============================] - 0s 218us/step - loss: 1135.2378 - acc: 0.3750
    
    Games Elapsed: 3192
    Current Epsilon: 46.90 percent
    
    Total Training duration: 4.27 minutes
    Current Iteration duration: 9.13 seconds
    Estimated Time remaining: 3.04 minutes
    
    ITERATION: 32 out of 50
    Player 1 Agent...
    Epoch 1/4
    66/66 [==============================] - 0s 120us/step - loss: 1014.9650 - acc: 0.3939
    Epoch 2/4
    66/66 [==============================] - 0s 121us/step - loss: 1000.7995 - acc: 0.4242
    Epoch 3/4
    66/66 [==============================] - 0s 105us/step - loss: 956.3792 - acc: 0.4394
    Epoch 4/4
    66/66 [==============================] - 0s 121us/step - loss: 932.3812 - acc: 0.4545
    Player 2 Agent...
    Epoch 1/4
    66/66 [==============================] - 0s 135us/step - loss: 1042.1895 - acc: 0.2879
    Epoch 2/4
    66/66 [==============================] - 0s 121us/step - loss: 1027.6189 - acc: 0.2727
    Epoch 3/4
    66/66 [==============================] - 0s 121us/step - loss: 1006.0515 - acc: 0.2727
    Epoch 4/4
    66/66 [==============================] - 0s 121us/step - loss: 987.3516 - acc: 0.2727
    
    Games Elapsed: 3292
    Current Epsilon: 46.80 percent
    
    Total Training duration: 4.40 minutes
    Current Iteration duration: 7.86 seconds
    Estimated Time remaining: 2.49 minutes
    
    ITERATION: 33 out of 50
    Player 1 Agent...
    Epoch 1/4
    68/68 [==============================] - 0s 117us/step - loss: 998.8995 - acc: 0.3088
    Epoch 2/4
    68/68 [==============================] - 0s 102us/step - loss: 967.3798 - acc: 0.3088
    Epoch 3/4
    68/68 [==============================] - 0s 117us/step - loss: 923.7579 - acc: 0.3382
    Epoch 4/4
    68/68 [==============================] - 0s 132us/step - loss: 892.3055 - acc: 0.3529
    Player 2 Agent...
    Epoch 1/4
    68/68 [==============================] - 0s 147us/step - loss: 1355.7656 - acc: 0.4118
    Epoch 2/4
    68/68 [==============================] - 0s 147us/step - loss: 1328.0192 - acc: 0.4265
    Epoch 3/4
    68/68 [==============================] - 0s 117us/step - loss: 1293.2691 - acc: 0.4265
    Epoch 4/4
    68/68 [==============================] - 0s 132us/step - loss: 1253.5679 - acc: 0.4706
    
    Games Elapsed: 3391
    Current Epsilon: 46.70 percent
    
    Total Training duration: 4.53 minutes
    Current Iteration duration: 7.83 seconds
    Estimated Time remaining: 2.35 minutes
    
    ITERATION: 34 out of 50
    Player 1 Agent...
    Epoch 1/4
    70/70 [==============================] - 0s 114us/step - loss: 830.7876 - acc: 0.3286
    Epoch 2/4
    70/70 [==============================] - 0s 128us/step - loss: 812.3865 - acc: 0.3429
    Epoch 3/4
    70/70 [==============================] - 0s 214us/step - loss: 786.1779 - acc: 0.3286
    Epoch 4/4
    70/70 [==============================] - 0s 114us/step - loss: 757.6480 - acc: 0.3286
    Player 2 Agent...
    Epoch 1/4
    70/70 [==============================] - 0s 142us/step - loss: 1825.7128 - acc: 0.3429
    Epoch 2/4
    70/70 [==============================] - 0s 214us/step - loss: 1744.4632 - acc: 0.3714
    Epoch 3/4
    70/70 [==============================] - 0s 128us/step - loss: 1670.7081 - acc: 0.3714
    Epoch 4/4
    70/70 [==============================] - 0s 142us/step - loss: 1586.1423 - acc: 0.4000
    
    Games Elapsed: 3490
    Current Epsilon: 46.60 percent
    
    Total Training duration: 4.66 minutes
    Current Iteration duration: 7.83 seconds
    Estimated Time remaining: 2.22 minutes
    
    ITERATION: 35 out of 50
    Player 1 Agent...
    Epoch 1/4
    72/72 [==============================] - 0s 110us/step - loss: 874.8211 - acc: 0.3056
    Epoch 2/4
    72/72 [==============================] - 0s 97us/step - loss: 862.4143 - acc: 0.3194
    Epoch 3/4
    72/72 [==============================] - 0s 97us/step - loss: 845.5999 - acc: 0.3333
    Epoch 4/4
    72/72 [==============================] - 0s 194us/step - loss: 829.6216 - acc: 0.3611
    Player 2 Agent...
    Epoch 1/4
    72/72 [==============================] - 0s 110us/step - loss: 1212.3541 - acc: 0.3750
    Epoch 2/4
    72/72 [==============================] - 0s 125us/step - loss: 1102.7044 - acc: 0.3889
    Epoch 3/4
    72/72 [==============================] - 0s 125us/step - loss: 976.0981 - acc: 0.4028
    Epoch 4/4
    72/72 [==============================] - 0s 97us/step - loss: 923.9899 - acc: 0.3889
    
    Games Elapsed: 3589
    Current Epsilon: 46.50 percent
    
    Total Training duration: 4.80 minutes
    Current Iteration duration: 8.29 seconds
    Estimated Time remaining: 2.21 minutes
    
    ITERATION: 36 out of 50
    Player 1 Agent...
    Epoch 1/4
    74/74 [==============================] - 0s 120us/step - loss: 1179.5519 - acc: 0.4054
    Epoch 2/4
    74/74 [==============================] - 0s 190us/step - loss: 1159.6278 - acc: 0.4054
    Epoch 3/4
    74/74 [==============================] - 0s 121us/step - loss: 1125.1880 - acc: 0.4324
    Epoch 4/4
    74/74 [==============================] - 0s 175us/step - loss: 1082.6878 - acc: 0.4459
    Player 2 Agent...
    Epoch 1/4
    74/74 [==============================] - 0s 189us/step - loss: 1470.7482 - acc: 0.4459
    Epoch 2/4
    74/74 [==============================] - 0s 162us/step - loss: 1401.6815 - acc: 0.4324
    Epoch 3/4
    74/74 [==============================] - 0s 135us/step - loss: 1319.4582 - acc: 0.4324
    Epoch 4/4
    74/74 [==============================] - 0s 189us/step - loss: 1240.2362 - acc: 0.4189
    
    Games Elapsed: 3689
    Current Epsilon: 46.40 percent
    
    Total Training duration: 4.95 minutes
    Current Iteration duration: 9.06 seconds
    Estimated Time remaining: 2.26 minutes
    
    ITERATION: 37 out of 50
    Player 1 Agent...
    Epoch 1/4
    76/76 [==============================] - 0s 118us/step - loss: 756.1344 - acc: 0.2895
    Epoch 2/4
    76/76 [==============================] - 0s 92us/step - loss: 741.5409 - acc: 0.2895
    Epoch 3/4
    76/76 [==============================] - 0s 118us/step - loss: 714.9023 - acc: 0.2895
    Epoch 4/4
    76/76 [==============================] - 0s 171us/step - loss: 685.9379 - acc: 0.2895
    Player 2 Agent...
    Epoch 1/4
    76/76 [==============================] - 0s 118us/step - loss: 963.4387 - acc: 0.3816
    Epoch 2/4
    76/76 [==============================] - 0s 170us/step - loss: 952.1915 - acc: 0.3684
    Epoch 3/4
    76/76 [==============================] - 0s 184us/step - loss: 929.2921 - acc: 0.3816
    Epoch 4/4
    76/76 [==============================] - 0s 144us/step - loss: 901.2061 - acc: 0.3684
    
    Games Elapsed: 3788
    Current Epsilon: 46.30 percent
    
    Total Training duration: 5.08 minutes
    Current Iteration duration: 7.89 seconds
    Estimated Time remaining: 1.84 minutes
    
    ITERATION: 38 out of 50
    Player 1 Agent...
    Epoch 1/4
    78/78 [==============================] - 0s 115us/step - loss: 1218.5587 - acc: 0.3846
    Epoch 2/4
    78/78 [==============================] - 0s 89us/step - loss: 1209.0824 - acc: 0.3846
    Epoch 3/4
    78/78 [==============================] - 0s 115us/step - loss: 1188.6717 - acc: 0.3846
    Epoch 4/4
    78/78 [==============================] - 0s 166us/step - loss: 1169.7608 - acc: 0.4103
    Player 2 Agent...
    Epoch 1/4
    78/78 [==============================] - 0s 115us/step - loss: 1091.9975 - acc: 0.3846
    Epoch 2/4
    78/78 [==============================] - 0s 153us/step - loss: 1048.2153 - acc: 0.3846
    Epoch 3/4
    78/78 [==============================] - 0s 166us/step - loss: 998.2100 - acc: 0.3974
    Epoch 4/4
    78/78 [==============================] - 0s 192us/step - loss: 935.5502 - acc: 0.3974
    
    Games Elapsed: 3888
    Current Epsilon: 46.20 percent
    
    Total Training duration: 5.21 minutes
    Current Iteration duration: 7.52 seconds
    Estimated Time remaining: 1.63 minutes
    
    ITERATION: 39 out of 50
    Player 1 Agent...
    Epoch 1/4
    80/80 [==============================] - 0s 112us/step - loss: 1016.7431 - acc: 0.3625
    Epoch 2/4
    80/80 [==============================] - 0s 100us/step - loss: 995.9139 - acc: 0.3625
    Epoch 3/4
    80/80 [==============================] - 0s 112us/step - loss: 970.0303 - acc: 0.3750
    Epoch 4/4
    80/80 [==============================] - 0s 137us/step - loss: 935.7877 - acc: 0.3875
    Player 2 Agent...
    Epoch 1/4
    80/80 [==============================] - 0s 162us/step - loss: 1333.7279 - acc: 0.4125
    Epoch 2/4
    80/80 [==============================] - 0s 112us/step - loss: 1287.0899 - acc: 0.4250
    Epoch 3/4
    80/80 [==============================] - 0s 149us/step - loss: 1240.0648 - acc: 0.4750
    Epoch 4/4
    80/80 [==============================] - 0s 162us/step - loss: 1193.3813 - acc: 0.4625
    
    Games Elapsed: 3988
    Current Epsilon: 46.10 percent
    
    Total Training duration: 5.35 minutes
    Current Iteration duration: 8.42 seconds
    Estimated Time remaining: 1.68 minutes
    
    ITERATION: 40 out of 50
    Player 1 Agent...
    Epoch 1/4
    82/82 [==============================] - 0s 109us/step - loss: 1740.0558 - acc: 0.2317
    Epoch 2/4
    82/82 [==============================] - 0s 122us/step - loss: 1705.5740 - acc: 0.2439
    Epoch 3/4
    82/82 [==============================] - 0s 195us/step - loss: 1645.2429 - acc: 0.2317
    Epoch 4/4
    82/82 [==============================] - 0s 122us/step - loss: 1593.1482 - acc: 0.2805
    Player 2 Agent...
    Epoch 1/4
    82/82 [==============================] - 0s 182us/step - loss: 1139.3692 - acc: 0.3049
    Epoch 2/4
    82/82 [==============================] - 0s 182us/step - loss: 1082.5516 - acc: 0.3049
    Epoch 3/4
    82/82 [==============================] - 0s 195us/step - loss: 1003.8335 - acc: 0.3171
    Epoch 4/4
    82/82 [==============================] - 0s 170us/step - loss: 947.4554 - acc: 0.3049
    
    Games Elapsed: 4087
    Current Epsilon: 46.00 percent
    
    Total Training duration: 5.49 minutes
    Current Iteration duration: 8.13 seconds
    Estimated Time remaining: 1.49 minutes
    
    ITERATION: 41 out of 50
    Player 1 Agent...
    Epoch 1/4
    84/84 [==============================] - 0s 118us/step - loss: 1269.5144 - acc: 0.2857
    Epoch 2/4
    84/84 [==============================] - 0s 119us/step - loss: 1238.0606 - acc: 0.3333
    Epoch 3/4
    84/84 [==============================] - 0s 166us/step - loss: 1196.4022 - acc: 0.3333
    Epoch 4/4
    84/84 [==============================] - 0s 131us/step - loss: 1134.4779 - acc: 0.3214
    Player 2 Agent...
    Epoch 1/4
    84/84 [==============================] - 0s 131us/step - loss: 1233.3200 - acc: 0.3095
    Epoch 2/4
    84/84 [==============================] - 0s 178us/step - loss: 1214.9672 - acc: 0.3095
    Epoch 3/4
    84/84 [==============================] - 0s 166us/step - loss: 1179.1174 - acc: 0.2976
    Epoch 4/4
    84/84 [==============================] - 0s 202us/step - loss: 1145.8901 - acc: 0.2976
    
    Games Elapsed: 4185
    Current Epsilon: 45.90 percent
    
    Total Training duration: 5.63 minutes
    Current Iteration duration: 8.65 seconds
    Estimated Time remaining: 1.44 minutes
    
    ITERATION: 42 out of 50
    Player 1 Agent...
    Epoch 1/4
    86/86 [==============================] - 0s 128us/step - loss: 1752.3671 - acc: 0.2674
    Epoch 2/4
    86/86 [==============================] - 0s 128us/step - loss: 1625.5194 - acc: 0.2674
    Epoch 3/4
    86/86 [==============================] - 0s 116us/step - loss: 1522.3146 - acc: 0.3023
    Epoch 4/4
    86/86 [==============================] - 0s 116us/step - loss: 1463.4052 - acc: 0.3721
    Player 2 Agent...
    Epoch 1/4
    86/86 [==============================] - 0s 104us/step - loss: 1617.4192 - acc: 0.2674
    Epoch 2/4
    86/86 [==============================] - 0s 116us/step - loss: 1591.7644 - acc: 0.2674
    Epoch 3/4
    86/86 [==============================] - 0s 116us/step - loss: 1544.3955 - acc: 0.2442
    Epoch 4/4
    86/86 [==============================] - 0s 104us/step - loss: 1516.2383 - acc: 0.2209
    
    Games Elapsed: 4285
    Current Epsilon: 45.80 percent
    
    Total Training duration: 5.77 minutes
    Current Iteration duration: 8.33 seconds
    Estimated Time remaining: 1.25 minutes
    
    ITERATION: 43 out of 50
    Player 1 Agent...
    Epoch 1/4
    88/88 [==============================] - 0s 113us/step - loss: 1353.5774 - acc: 0.2727
    Epoch 2/4
    88/88 [==============================] - 0s 91us/step - loss: 1226.9423 - acc: 0.2955
    Epoch 3/4
    88/88 [==============================] - 0s 102us/step - loss: 1138.5354 - acc: 0.3295
    Epoch 4/4
    88/88 [==============================] - 0s 102us/step - loss: 1068.9423 - acc: 0.3295
    Player 2 Agent...
    Epoch 1/4
    88/88 [==============================] - 0s 102us/step - loss: 1470.0378 - acc: 0.2727
    Epoch 2/4
    88/88 [==============================] - 0s 90us/step - loss: 1411.2333 - acc: 0.3068
    Epoch 3/4
    88/88 [==============================] - 0s 102us/step - loss: 1323.6924 - acc: 0.3182
    Epoch 4/4
    88/88 [==============================] - 0s 102us/step - loss: 1229.4121 - acc: 0.3636
    
    Games Elapsed: 4385
    Current Epsilon: 45.70 percent
    
    Total Training duration: 5.90 minutes
    Current Iteration duration: 7.96 seconds
    Estimated Time remaining: 1.06 minutes
    
    ITERATION: 44 out of 50
    Player 1 Agent...
    Epoch 1/4
    90/90 [==============================] - 0s 110us/step - loss: 1309.0655 - acc: 0.3333
    Epoch 2/4
    90/90 [==============================] - 0s 177us/step - loss: 1266.8608 - acc: 0.3222
    Epoch 3/4
    90/90 [==============================] - 0s 177us/step - loss: 1209.5399 - acc: 0.3222
    Epoch 4/4
    90/90 [==============================] - 0s 134us/step - loss: 1158.6203 - acc: 0.3222
    Player 2 Agent...
    Epoch 1/4
    90/90 [==============================] - 0s 110us/step - loss: 1144.2557 - acc: 0.3222
    Epoch 2/4
    90/90 [==============================] - 0s 166us/step - loss: 1143.7764 - acc: 0.3333
    Epoch 3/4
    90/90 [==============================] - 0s 177us/step - loss: 1116.4928 - acc: 0.3444
    Epoch 4/4
    90/90 [==============================] - 0s 166us/step - loss: 1088.5306 - acc: 0.3778
    
    Games Elapsed: 4485
    Current Epsilon: 45.60 percent
    
    Total Training duration: 6.05 minutes
    Current Iteration duration: 8.65 seconds
    Estimated Time remaining: 1.01 minutes
    
    ITERATION: 45 out of 50
    Player 1 Agent...
    Epoch 1/4
    92/92 [==============================] - 0s 98us/step - loss: 1002.3392 - acc: 0.3152
    Epoch 2/4
    92/92 [==============================] - 0s 163us/step - loss: 995.5119 - acc: 0.3261
    Epoch 3/4
    92/92 [==============================] - 0s 152us/step - loss: 978.6890 - acc: 0.3370
    Epoch 4/4
    92/92 [==============================] - 0s 108us/step - loss: 965.4399 - acc: 0.3478
    Player 2 Agent...
    Epoch 1/4
    92/92 [==============================] - 0s 163us/step - loss: 1062.0956 - acc: 0.3043
    Epoch 2/4
    92/92 [==============================] - 0s 162us/step - loss: 1033.3203 - acc: 0.3043
    Epoch 3/4
    92/92 [==============================] - 0s 173us/step - loss: 994.9631 - acc: 0.3043
    Epoch 4/4
    92/92 [==============================] - 0s 173us/step - loss: 958.0337 - acc: 0.3152
    
    Games Elapsed: 4584
    Current Epsilon: 45.50 percent
    
    Total Training duration: 6.19 minutes
    Current Iteration duration: 8.53 seconds
    Estimated Time remaining: 0.85 minutes
    
    ITERATION: 46 out of 50
    Player 1 Agent...
    Epoch 1/4
    94/94 [==============================] - 0s 95us/step - loss: 1002.3621 - acc: 0.2766
    Epoch 2/4
    94/94 [==============================] - 0s 151us/step - loss: 993.2285 - acc: 0.2766
    Epoch 3/4
    94/94 [==============================] - 0s 106us/step - loss: 978.9589 - acc: 0.3298
    Epoch 4/4
    94/94 [==============================] - 0s 116us/step - loss: 966.0448 - acc: 0.3298
    Player 2 Agent...
    Epoch 1/4
    94/94 [==============================] - 0s 149us/step - loss: 1295.5495 - acc: 0.3830
    Epoch 2/4
    94/94 [==============================] - 0s 170us/step - loss: 1263.0013 - acc: 0.3936
    Epoch 3/4
    94/94 [==============================] - 0s 170us/step - loss: 1199.9671 - acc: 0.4043
    Epoch 4/4
    94/94 [==============================] - 0s 139us/step - loss: 1151.7080 - acc: 0.4149
    
    Games Elapsed: 4682
    Current Epsilon: 45.40 percent
    
    Total Training duration: 6.31 minutes
    Current Iteration duration: 7.44 seconds
    Estimated Time remaining: 0.62 minutes
    
    ITERATION: 47 out of 50
    Player 1 Agent...
    Epoch 1/4
    96/96 [==============================] - 0s 94us/step - loss: 811.2246 - acc: 0.3333
    Epoch 2/4
    96/96 [==============================] - 0s 104us/step - loss: 797.4875 - acc: 0.3646
    Epoch 3/4
    96/96 [==============================] - 0s 104us/step - loss: 778.7279 - acc: 0.3542
    Epoch 4/4
    96/96 [==============================] - 0s 103us/step - loss: 749.3211 - acc: 0.3854
    Player 2 Agent...
    Epoch 1/4
    96/96 [==============================] - 0s 135us/step - loss: 634.0662 - acc: 0.3333
    Epoch 2/4
    96/96 [==============================] - 0s 143us/step - loss: 620.9817 - acc: 0.3229
    Epoch 3/4
    96/96 [==============================] - 0s 104us/step - loss: 601.9964 - acc: 0.3438
    Epoch 4/4
    96/96 [==============================] - 0s 104us/step - loss: 584.1803 - acc: 0.3333
    
    Games Elapsed: 4781
    Current Epsilon: 45.30 percent
    
    Total Training duration: 6.46 minutes
    Current Iteration duration: 9.18 seconds
    Estimated Time remaining: 0.61 minutes
    
    ITERATION: 48 out of 50
    Player 1 Agent...
    Epoch 1/4
    98/98 [==============================] - 0s 102us/step - loss: 1219.8431 - acc: 0.3367
    Epoch 2/4
    98/98 [==============================] - 0s 102us/step - loss: 1191.3309 - acc: 0.3571
    Epoch 3/4
    98/98 [==============================] - 0s 142us/step - loss: 1157.3705 - acc: 0.3469
    Epoch 4/4
    98/98 [==============================] - 0s 122us/step - loss: 1130.5971 - acc: 0.3469
    Player 2 Agent...
    Epoch 1/4
    98/98 [==============================] - 0s 122us/step - loss: 1290.4189 - acc: 0.3776
    Epoch 2/4
    98/98 [==============================] - 0s 102us/step - loss: 1274.3649 - acc: 0.4082
    Epoch 3/4
    98/98 [==============================] - 0s 112us/step - loss: 1253.0000 - acc: 0.3878
    Epoch 4/4
    98/98 [==============================] - 0s 102us/step - loss: 1229.5792 - acc: 0.3878
    
    Games Elapsed: 4880
    Current Epsilon: 45.20 percent
    
    Total Training duration: 6.60 minutes
    Current Iteration duration: 8.22 seconds
    Estimated Time remaining: 0.41 minutes
    
    ITERATION: 49 out of 50
    Player 1 Agent...
    Epoch 1/4
    100/100 [==============================] - 0s 100us/step - loss: 1718.0331 - acc: 0.3500
    Epoch 2/4
    100/100 [==============================] - 0s 109us/step - loss: 1636.5534 - acc: 0.3200
    Epoch 3/4
    100/100 [==============================] - 0s 150us/step - loss: 1511.5339 - acc: 0.3500
    Epoch 4/4
    100/100 [==============================] - 0s 190us/step - loss: 1436.5570 - acc: 0.3700
    Player 2 Agent...
    Epoch 1/4
    100/100 [==============================] - 0s 150us/step - loss: 1140.0796 - acc: 0.4300
    Epoch 2/4
    100/100 [==============================] - 0s 160us/step - loss: 1116.5595 - acc: 0.4200
    Epoch 3/4
    100/100 [==============================] - 0s 120us/step - loss: 1090.9453 - acc: 0.4200
    Epoch 4/4
    100/100 [==============================] - 0s 170us/step - loss: 1066.4120 - acc: 0.4300
    
    Games Elapsed: 4980
    Current Epsilon: 45.10 percent
    
    Total Training duration: 6.76 minutes
    Current Iteration duration: 9.57 seconds
    Estimated Time remaining: 0.32 minutes
    
    ITERATION: 50 out of 50
    Player 1 Agent...
    Epoch 1/4
    100/100 [==============================] - 0s 89us/step - loss: 1184.5830 - acc: 0.2400
    Epoch 2/4
    100/100 [==============================] - 0s 110us/step - loss: 1067.1543 - acc: 0.2400
    Epoch 3/4
    100/100 [==============================] - 0s 130us/step - loss: 995.0295 - acc: 0.2600
    Epoch 4/4
    100/100 [==============================] - 0s 140us/step - loss: 960.7215 - acc: 0.2500
    Player 2 Agent...
    Epoch 1/4
    100/100 [==============================] - 0s 169us/step - loss: 687.8488 - acc: 0.3300
    Epoch 2/4
    100/100 [==============================] - 0s 170us/step - loss: 647.2836 - acc: 0.3300
    Epoch 3/4
    100/100 [==============================] - 0s 170us/step - loss: 611.3624 - acc: 0.3500
    Epoch 4/4
    100/100 [==============================] - 0s 160us/step - loss: 584.2605 - acc: 0.3300
    
    Games Elapsed: 5080
    Current Epsilon: 45.00 percent
    
    Total Training duration: 6.89 minutes
    Current Iteration duration: 7.78 seconds
    Estimated Time remaining: 0.13 minutes
    


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
    
