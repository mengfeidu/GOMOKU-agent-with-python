import random
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
from evaluate import*
import copy
import time

pp.infotext = 'name="pbrain-pyrandom", author="Jan Stransky", version="1.0", country="Czech Republic", www="https://github.com/stranskyjan/pbrain-pyrandom"'

MAX_BOARD = 100
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]

THRESHOLD = 3
inBoundary = lambda x, y: (x >= 0 and y >= 0 and x < pp.width and y < pp.height)


# class pptest:
#     def __init__(self, board):
#         self.width = len(board)
#         self.height = len(board[0])
#         # self.lt = board
#         # self.set = set1
# board = [[0 for _ in range(20)] for _ in range(20)]
# pp = pptest(board)


def brain_init():
    if pp.width < 5 or pp.height < 5:
        pp.pipeOut("ERROR size of the board")
        return
    if pp.width > MAX_BOARD or pp.height > MAX_BOARD:
        pp.pipeOut("ERROR Maximal board size is {}".format(MAX_BOARD))
        return
    pp.pipeOut("OK")
def brain_restart():
    for x in range(pp.width):
        for y in range(pp.height):
            board[x][y] = 0
    pp.pipeOut("OK")
def isFree(x, y):
    return x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] == 0
def brain_my(x, y):
    if isFree(x, y):
        board[x][y] = 1
    else:
        pp.pipeOut("ERROR my move [{},{}]".format(x, y))
def brain_opponents(x, y):
    if isFree(x, y):
        board[x][y] = 2
    else:
        pp.pipeOut("ERROR opponents's move [{},{}]".format(x, y))
def brain_block(x, y):
    if isFree(x, y):
        board[x][y] = 3
    else:
        pp.pipeOut("ERROR winning move [{},{}]".format(x, y))
def brain_takeback(x, y):
    if x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] != 0:
        board[x][y] = 0
        return 0
    return 2
def brain_turn():
    if pp.terminateAI:
        return
    i = 0
    while True:
        x,y = Mybrain_turn()
        i += 1
        if pp.terminateAI:
            return
        if isFree(x, y):
            break
    if i > 1:
        pp.pipeOut("DEBUG {} coordinates didn't hit an empty field".format(i))
    pp.do_mymove(x, y)




def Mybrain_turn():

    player = 1  # 假设是我方轮次
    best_position = None
    best_position = get_successor(board,player)
    x, y = best_position

    return best_position
    #pp.do_mymove(x,y)


def nearKsquares(K, board):
    # return a list 扫一遍
    NearNeighbor = dict()
    allposition = 0
    for x in range(pp.width):
        for y in range(pp.height):
            if board[x][y] == 0:
                neighbor_num = 0
                for x_might in range(max(x - K, 0), min(pp.width, x + K + 1)):
                    for y_might in range(max(y - K, 0), min(pp.height, y + K + 1)):
                        if board[x_might][y_might] != 0:  # 可能有点偏僻
                            neighbor_num += 1
                if neighbor_num > 0:
                    NearNeighbor[(x, y)] = neighbor_num
            else:
                allposition += 1

    if len(NearNeighbor) >= 3 and allposition >= 3:  # 缩小搜素范围
        NearNeighbor = dict((key, value) for key, value in NearNeighbor.items() if value > 1)

    NearNeighborlist = sorted(NearNeighbor.keys(), key=lambda x: NearNeighbor[x], reverse=True)

    if len(NearNeighbor) == 0:
        return [(int(pp.width / 2), int(pp.height / 2))]

    return NearNeighborlist

def if_win(board, player, x_last, y_last):
    # 遍历看四周是否能赢
    # 横着的路
    tmp = 0
    for x in range(max(x_last - 4, 0), min(pp.width, x_last + 5)):
        if board[x][y_last] == player:
            tmp += 1
        else:
            tmp = 0
            continue
        if tmp == 5:
            return True
    # 竖着的路
    tmp = 0
    for y in range(max(y_last - 4, 0), min(pp.height, y_last + 5)):
        if board[x_last][y] == player:
            tmp += 1
        else:
            tmp = 0
            continue
        if tmp == 5:
            return True
    # 东北-西南
    tmp = 0
    for y in range(max(y_last - 4, 0), min(pp.height, y_last + 5)):
        x = x_last + (y - y_last)
        if x >= 0 and y >= 0 and x < pp.width and y < pp.height:
            if board[x][y] == player:
                tmp += 1
            else:
                tmp = 0
        else:
            tmp = 0
        if tmp == 5:
            return True
    # 西北-东南
    tmp = 0
    for y in range(max(y_last - 4, 0), min(pp.height, y_last + 5)):
        x = x_last - (y - y_last)
        if x >= 0 and y >= 0 and x < pp.width and y < pp.height:
            if board[x][y] == player:
                tmp += 1
            else:
                tmp = 0
        else:
            tmp = 0
        if tmp == 5:
            return True

    return False

def get_successor(board, player):
    # return 当前player的successor
    mykill = static_evaluate(board, player)
    opponent = 3 - player
    mythreat = static_evaluate(board, opponent)

    successors = nearKsquares(2, board)
    max_value = -9999
    best_position = None
    for successor in successors:
        case = position_evaluate(board,successor,player)
        static_my = evaluation(mykill)
        static_op = evaluation(mythreat)
        if kill_prune(case,mythreat):
            return successor
        v = evaluation(case)
        if v + static_my - static_op > max_value:
            max_value = v + static_my - static_op
            best_position = successor

    return best_position

def kill_prune(case,mythreat):
    if case['WIN']> 0:
        return 1
    l4 = ['L4','S41','S42', 'S43', 'S44','S45']
    sum_op = 0
    for item in l4:
        sum_op += mythreat[item]
    if case['L4'] > 0 and sum_op == 0:
        return 1
    sum_my= 0
    for item in l4:
        sum_my += case[item]
    if sum_op == 0 and sum_my > 0 :
        return 1
    return 0



def free(x,y):
    return (x >= 0 and y >= 0 and x < pp.width and y < pp.height)

# **********************************************************************************
#   测试函数
# **********************************************************************************


def testalpha_beta():
    # 修改 board 即可得到相应值
    global  board
    board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]

    player = 1

    # print( get_successors(board,player=player) )


    best = Mybrain_turn()
    x , y =best
    board[x][y] = player
    print(x)
    print(y)
    print(best)
    if if_win(board, player, x, y):
        print('end game!!! a-b win')

    for i in range(len(board)):
        print(board[i])



# if __name__ == '__main__':
#     print('okk')
#
#     # game()
#     testalpha_beta()

def brain_end():
    pass


def brain_about():
    pp.pipeOut(pp.infotext)


if DEBUG_EVAL:
    import win32gui


    def brain_eval(x, y):
        # TODO check if it works as expected
        wnd = win32gui.GetForegroundWindow()
        dc = win32gui.GetDC(wnd)
        rc = win32gui.GetClientRect(wnd)
        c = str(board[x][y])
        win32gui.ExtTextOut(dc, rc[2] - 15, 3, 0, None, c, ())
        win32gui.ReleaseDC(wnd, dc)




######################################################################

# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = brain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about
if DEBUG_EVAL:
    pp.brain_eval = brain_eval


def main():
    pp.main()


if __name__ == "__main__":
    main()