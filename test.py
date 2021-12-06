import numpy as np
import random

qTable = {}
representStates = [0, 1, 2]

def getHashValue(hash):
  if not hash in qTable:
    qTable[hash] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

  return qTable[hash]
  

def updateHash(hash, newValue):
  qTable[hash] = newValue

def getPossibilityActions(hash):
  possibilityActions = []
  for stringValue in hash:
    value = int(stringValue)
    if value != 0:
      possibilityActions.append(0)
    else:
      possibilityActions.append(1)
  return np.array(possibilityActions)

def stateToHash(state):
  hash = ""
  for s in state:
    hash += str(int(s))
  return hash

class Agent:
  def __init__(self, epsilon=0.3, lr=0.3, gamma=.99, isPlay=False):
    self.epsilon = epsilon
    self.lr = lr
    self.gamma = gamma
    self.isPlay = isPlay

  def act(self, state):
    rand = random.uniform(0, 1)
    # convert state to hash
    hash = stateToHash(state)

    # get possibility actions
    possibilityActions = getPossibilityActions(hash)

    # get Q value
    qValues = getHashValue(hash)

    # random Q value
    if rand < self.epsilon and not self.isPlay:
      qValues = np.random.rand(9)
    
    # avoid choice same action when qValue is negative
    qValues = np.array(qValues)
    if qValues.min() < 0:
      base = abs(qValues.min())
      qValues += base * 2

    # dot product
    qValues = np.multiply(qValues, possibilityActions)

    # avoid use first action when nothing to choose
    if qValues.sum() == 0:
      qValues = possibilityActions

    # random if have multiple best action
    if np.count_nonzero(qValues == qValues.max()) > 1:
      bestActions = [i for i in range(len(qValues)) if qValues[i] == qValues.max()]
      return random.choice(bestActions)

    # print(qValues)
    # choose best action
    return np.argmax(qValues)

  def learn(self, state, nextState, action, reward, isDone):
    hashState = stateToHash(state)
    hashNextState = stateToHash(nextState)

    qState = getHashValue(hashState)
    qNextState = getHashValue(hashNextState)

    possibilityActions = getPossibilityActions(hashNextState)
    qNextState = np.multiply(qNextState, possibilityActions)

    tmpQNextState = np.array(qNextState, copy=True)
    if qNextState.min() < 0:
      base = abs(qNextState.min())
      tmpQNextState += base * 2

    qState[action] += self.lr * (reward + self.gamma * qNextState[np.argmax(tmpQNextState)] - qState[action])
    if isDone:
      qState[action] = reward

    updateHash(hashState, qState)
  
class Env:
  def __init__(self):
    self.reset()

  def reset(self):
    self.board = np.zeros((9,))
    self.isXTurn = True
    return self.getState()

  def checkRows(self, board):
    for row in board:
        if len(set(row)) == 1:
            return row[0]
    return 0

  def checkDiagonals(self, board):
    if len(set([board[i][i] for i in range(len(board))])) == 1:
        return board[0][0]
    if len(set([board[i][len(board)-i-1] for i in range(len(board))])) == 1:
        return board[0][len(board)-1]
    return 0

  def checkWin(self):
    board = self.board.reshape((3,3))
    for newBoard in [board, np.transpose(board)]:
        result = self.checkRows(newBoard)
        if result:
            return result
    return self.checkDiagonals(board)

  def checkDraw(self):
    return self.checkWin() == 0

  def checkDone(self):
    return self.board.min() != 0 or self.checkWin() != 0

  def getState(self):
    return np.array(self.board, copy=True)

  def showBoard(self):
    prettyBoard = self.board.reshape((3, 3))
    for row in prettyBoard:
      print("|", end='')
      for col in row:
        symbol = "*"
        if col == 1:
          symbol = "X"
        elif col == 2:
          symbol = "O"
        print(symbol, end='')
        print("|", end='')
      print("")

  def act(self, action):
    reward = 0
    player = 2
    if self.isXTurn:
      player = 1

    self.board[action] = player
    self.isXTurn = not self.isXTurn

    winner = self.checkWin()
    isDraw = self.checkDraw()
    isDone = self.checkDone()

    if winner:
      reward = 1
    
    if isDraw:
      reward = 0.5

    nextState = np.array(self.board, copy=True)
    return nextState, reward, isDone, {}


env = Env()
agent = Agent()

env.getState()
# print(env.getState())

episodes = 2000
winner_history = []

def swapSide(state):
  newState = np.array(state, copy=True)

  for i in range(len(newState)):
    if newState[i] == 1:
      newState[i] = 2
    elif newState[i] == 2:
      newState[i] = 1

  return newState

def rotage(state, n = 1):
  return np.rot90(state.reshape((3,3)), n).reshape((9,))


def rotageAction(action, n = 1):
  board = np.zeros((9,))
  board[action] = 1
  board = rotage(board, n)
  return np.argmax(board)

for episode in range(episodes):
  isDone = False
  state = env.reset()
  prevState = state
  prevAction = -1
  isShouldLearn = False
  
  if episode % 1000 == 0:
    print("episode:", episode)

  while not isDone:
    state = env.getState()

    if not env.isXTurn:
      state = swapSide(state)
    
    action = agent.act(state)
    nextState, reward, isDone, _ = env.act(action)
    # env.showBoard()

    # if X turn mean before act is not X turn
    if env.isXTurn:
      nextState = swapSide(nextState)

    if isShouldLearn:
      if isDone and not env.checkDraw():
        prevReward = -1
      elif isDone and env.checkDraw():
        prevReward = 0.5
      agent.learn(prevState, swapSide(nextState), prevAction, prevReward, isDone)
      agent.learn(rotage(prevState, 1), rotage(swapSide(nextState), 1), rotageAction(prevAction, 1), prevReward, isDone)
      agent.learn(rotage(prevState, 2), rotage(swapSide(nextState), 2), rotageAction(prevAction, 2), prevReward, isDone)
      agent.learn(rotage(prevState, 3), rotage(swapSide(nextState), 3), rotageAction(prevAction, 3), prevReward, isDone)
      

    if isDone:
      agent.learn(state, nextState, action, reward, isDone)
      agent.learn(rotage(state, 1), rotage(nextState, 1), rotageAction(action, 1), reward, isDone)
      agent.learn(rotage(state, 2), rotage(nextState, 2), rotageAction(action, 2), reward, isDone)
      agent.learn(rotage(state, 3), rotage(nextState, 3), rotageAction(action, 3), reward, isDone)

    prevState = state
    prevAction = action
    prevReward = reward
    isShouldLearn = True

  winner_history.append(env.checkWin())

len(qTable)

class TigTagToeGame:
  def __init__(self):
    self.reset()

  def reset(self):
    self.board = np.zeros((9,))
    self.isXTurn = True
    return self.getState()

  def checkRows(self, board):
    for row in board:
        if len(set(row)) == 1:
            return row[0]
    return 0

  def checkDiagonals(self, board):
    if len(set([board[i][i] for i in range(len(board))])) == 1:
        return board[0][0]
    if len(set([board[i][len(board)-i-1] for i in range(len(board))])) == 1:
        return board[0][len(board)-1]
    return 0

  def checkWin(self):
    board = self.board.reshape((3,3))
    for newBoard in [board, np.transpose(board)]:
        result = self.checkRows(newBoard)
        if result:
            return result
    return self.checkDiagonals(board)

  def checkDraw(self):
    return self.checkWin() == 0

  def checkDone(self):
    return self.board.min() != 0 or self.checkWin() != 0

  def getState(self):
    return np.array(self.board, copy=True)

  def showBoard(self):
    prettyBoard = self.board.reshape((3, 3))
    for row in prettyBoard:
      print("|", end='')
      for col in row:
        symbol = "*"
        if col == 1:
          symbol = "X"
        elif col == 2:
          symbol = "O"
        print(symbol, end='')
        print("|", end='')
      print("")


  def play(self, action):
    player = 2
    if self.isXTurn:
      player = 1

    self.board[action] = player
    self.isXTurn = not self.isXTurn

    winner = self.checkWin()
    isDone = self.checkDone()

    nextState = np.array(self.board, copy=True)
    return nextState, isDone

game = TigTagToeGame()
agent = Agent(isPlay=True)

game.showBoard()

isDone = False
game.reset()

while not isDone:
  state = game.getState()
  print("--- AI vs Human ---")
  game.showBoard()

  action = 0
  if game.isXTurn:
    action = agent.act(state)
    isInputValidate = False
    while not isInputValidate:
      action = int(input("player turn (X):"))
      if len(state) > action and state[action] == 0:
        isInputValidate = True
    print("thinking x", getHashValue(stateToHash(state)))
    if state[4] == 0:
      action = 4
  else:
    sstate = swapSide(state)
    print("thinking", getHashValue(stateToHash(sstate)))
    action = agent.act(swapSide(state))
  print(action)
  state, isDone = game.play(action)

print("game end")
game.showBoard()
winner = game.checkWin()
if winner == 1:
  print("Congratulation the player win.")
elif winner == 2:
  print("AI is the winner, We'll conquer the world")
else:
  print("Draw!!")