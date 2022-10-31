import gym
import math
import random
import requests
import numpy as np
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

RC = 6
CC = 7
PP = 1
AP = -1
WL = 4

SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = [""] # We wrote our student-id so that the school could log how well our implementation performed  

def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })

   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

"""
Trivial test agent provided to us so we could test that our implementation worked better than random placement before connecting to the server.  It returns a move 0-6 or -1 
if it could not make a move. To check your code for better performance, change this code to use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done

 #Places a piece on the board
def play_piece(board, row, col, piece):
    board[row][col] = piece

#Checks if the column has any valid rows left to place a piece in
def is_valid_location(board, col):
    return board[0][col] == 0

#Find the next open row in a certain column of the board
def next_row(board, col):
    for r in range(RC):
        nrc = RC - 1 - r
        if board[nrc][col] == 0:
            return nrc

#Check for any winning moves on the board
def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(CC - 3):
        for r in range(RC):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(CC):
        for r in range(RC - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True

    # Check diagonals
    for c in range(CC - 3):
        for r in range(RC - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True

    for c in range(CC - 3):
        for r in range(3, RC):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True

#Evaluates the windows containing moves created by score_position and then scores them
def evaluation(window):
    score = 0


    if window.count(1) == 4:
        score += 100
    elif window.count(1) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(1) == 2 and window.count(0) == 2:
        score += 2
    elif window.count(-1) == 3 and window.count(0) == 1:
        score -= 100
    elif window.count(-1) == 3 and window.count(0) == 1:
        score -= 4
    elif window.count(-1) == 2 and window.count(0) == 2:
        score -= 2

    return score

#Here i create multiple small windows of length four to check for what combinations there are on the board.
#I then send that window to the function evalutation to get the score for it.
def score_position(board, piece):
    score = 0

    ## Score center column
    center_array = [int(i) for i in list(board[:, 3])]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for r in range(RC):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(CC - 3):
            window = row_array[c:c + WL]
            score += evaluation(window)

    ## Score Vertical
    for c in range(CC):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(RC - 3):
            window = col_array[r:r + WL]
            score += evaluation(window)

    ## Score diagonals
    for r in range(RC - 3):
        for c in range(CC - 3):
            window = [board[r + i][c + i] for i in range(WL)]
            score += evaluation(window)

    for r in range(RC - 3):
        for c in range(CC - 3):
            window = [board[r + 3 - i][c + i] for i in range(WL)]
            score += evaluation(window)

    return score

#Checks if there are any winning moves for either me or the server or if the board is full and we have a draw.
def terminal_nodes(board):
    return winning_move(board, PP) or winning_move(board, AP) or len(get_valid_locations(board)) == 0

#Here i have implemented the minimax
def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = terminal_nodes(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, PP):
                return (None, 1000000000000000)
            elif winning_move(board, AP):
                return (None, -100000000000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, PP))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = next_row(board, col)
            board_copy = board.copy()
            play_piece(board_copy, row, col, PP)
            new_score = minimax(board_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = next_row(board, col)
            board_copy = board.copy()
            play_piece(board_copy, row, col, AP)
            new_score = minimax(board_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

#Finds all the valid locations to drop a piece and puts them into a list.
def get_valid_locations(board):
    valid_locations = []
    for col in range(CC):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

#My student_move which use to play the game with
def student_move(state):
    move, mm_score = minimax(state, 4, -math.inf, math.inf, True)
    return move




def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a botaction = random.choice(list(avmoves)) reward from the server after each move, but it is 0 while the game is running
   loss = -1, win = +1, draw = +0.5, error = -10 (you get this if you try to play in a full column). Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(state) # TODO: change input here

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
         env.reset(state)
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! Games ends.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def main():
   play_game(vs_server = True)

if __name__ == "__main__":
    main()
