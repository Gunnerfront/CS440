import math

import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def argmax(vals):
  currMaxIdx = 0
  for idx in range(1, len(vals)):
    if vals[idx] > vals[currMaxIdx]:
      currMaxIdx = idx
  return currMaxIdx

def argmin(vals):
  currMinIdx = 0
  for idx in range(1, len(vals)):
    if vals[idx] < vals[currMinIdx]:
      currMinIdx = idx
  return currMinIdx

def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0:
      value = evaluate(board)
      return value, [], {}
    
    moveTree = {}
    moveVals = []
    moveLists = []
    for move in generateMoves(side,board,flags):
      newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])
      recurseVal, recurseList, recurseMoveTree = minimax(newSide, newBoard, newFlags, depth - 1)
      
      moveVals.append(recurseVal)
      recurseList.insert(0, move)
      moveLists.append(recurseList)
      moveTree[encode(*move)] = recurseMoveTree

    bestMove = 0
    if side:
      # if black (Min)
      bestMove = argmin(moveVals)
    else:
      # if white (Max)
      bestMove = argmax(moveVals)

    return moveVals[bestMove], moveLists[bestMove], moveTree
    # minimax

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0:
      value = evaluate(board)
      return value, [], {}
    
    moveTree = {}
    moveVals = []
    moveLists = []
    for move in generateMoves(side,board,flags):
      newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])
      recurseVal, recurseList, recurseMoveTree = alphabeta(newSide, newBoard, newFlags, depth - 1, alpha, beta)
      
      moveVals.append(recurseVal)
      recurseList.insert(0, move)
      moveLists.append(recurseList)
      moveTree[encode(*move)] = recurseMoveTree

      if side and beta > recurseVal:
        # if black (Min), updates beta
        beta = recurseVal
      if not side and alpha < recurseVal:
        # if white (Max) updates alpha
        alpha = recurseVal
      if alpha >= beta:
        # prune other children
        break

    bestMove = 0
    if side:
      # if black (Min)
      bestMove = argmin(moveVals)
    else:
      # if white (Max)
      bestMove = argmax(moveVals)

    return moveVals[bestMove], moveLists[bestMove], moveTree
    # alphabeta
    

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    if depth == 0:
      value = evaluate(board)
      return value, [], {}
    
    moveTree = {}
    moveVals = []
    moveLists = []
    for move in generateMoves(side,board,flags):
      newSide, newBoard, newFlags = makeMove(side, board, move[0], move[1], flags, move[2])
      recurseVal, recurseList, recurseMoveTree = stochastic(newSide, newBoard, newFlags, depth - 1, , )
      
      moveVals.append(recurseVal)
      recurseList.insert(0, move)
      moveLists.append(recurseList)
      moveTree[encode(*move)] = recurseMoveTree

    bestMove = 0
    if side:
      # if black (Min)
      bestMove = argmin(moveVals)
    else:
      # if white (Max)
      bestMove = argmax(moveVals)

    return moveVals[bestMove], moveLists[bestMove], moveTree
    # stochastic
