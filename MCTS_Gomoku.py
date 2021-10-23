import random
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
import math
import copy


pp.infotext = 'name="pbrain-pyrandom", author="Jan Stransky", version="1.0", country="Czech Republic", www="https://github.com/stranskyjan/pbrain-pyrandom"'

MAX_BOARD = 100
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]

THRESHOLD = 3
inBoundary = lambda x, y: (x >= 0 and y >= 0 and x < pp.width and y < pp.height)

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


# MCTS METHOD ***************************************************************** BEGIN
#  带有 ‘****’ 注释表明可以调节 / 可以
class MctNode:
    def __init__(self,parent,chess_position,player):
        self.parent = parent # 父节点
        self.N = 0 # 访问次数 （也即包含该落子状态的模拟对局次数）
        self.value = 0 # MCTS 计算得到的总分
        self.child = [] # 记录 expand 后的子节点
        self.potentialChild = [] # 记录潜在的可能子节点 (unexpanded)
        self.position = chess_position # 该步落子位置
        self.player = player # 当前落子位置对应玩家
        # ************************
        self.value_tune = 0   # 改进版

class MCTS:
    def __init__(self,board,player,tune = True):
        self.board = board
        self.player = player
        self.opponent = 3 - self.player
        self.root = MctNode(parent=None, chess_position=None ,player = 3 - self.player)
        self.tune = tune


    def UCTsearch(self,IterMAX=1000):
        # 主函数 直白返回结果
        origin_node = self.root # 初始根节点
        successors = get_successors(board=self.board, player = self.player) # 根节点的可能儿子 / 即为我方下一步的可能落子点
        if len(successors) == 1:
            return MctNode(parent=self.root , chess_position=successors[-1] ,player = 3 - self.player)
        origin_node.potentialChild = successors[0:min(len(successors) , 4)] # 缩小选择范围 / informed - suboptimal - converge ***************
        for _ in range(IterMAX):
            vnode = self.treePolicy(origin_node)
            reward = self.defualtpolicy(vnode.player)
            self.backup(vnode,reward)
        return self.bestchild(origin_node,0) # 直白得出结果

    def bestchild(self,parentNode,c=0):
        #  select the expanded child node with c = sqrt(2) ; select final target
        #  MctNode contains: parent,visit_num,chess_position,player
        maxi = -9999
        maxiNode = None
        for childNode in parentNode.child:
            if childNode.N == 0:  return childNode
            if not self.tune:
                # 原版 UCB
                Q =  childNode.value / childNode.N  + c *  math.sqrt( 2 * math.log(parentNode.N) / childNode.N )
            else:
                # tuned Q : 一点加速收敛小改进 // 可见实验
                V = 0.5 * childNode.value_tune - (childNode.value / childNode.N) ** 2 + math.sqrt(2 * math.log(parentNode.N) / childNode.N)
                Q = childNode.value / childNode.N + math.sqrt(math.log(parentNode.N) / childNode.N * min(0.25, V))
            if Q > maxi:
                maxiNode = childNode
                maxi = Q
            elif Q == maxi and random.uniform(0,1) > 0.5 : # 随机选择
                maxiNode = childNode
        return maxiNode

    def treePolicy(self,node):
        # node means v in reference
        # 此函数用于 expand 的步骤
        # n 层数目, ********************* 可调节

        TERMINATE_TIME = 9 # 此步是限制 treepolicy 的树高 *************** 可以调整

        for i in range(TERMINATE_TIME):
            if  (i > 0) and if_win(board = self.board ,player = node.player,
                      x_last = node.position[0], y_last = node.position[1]):
                return node
            else:
                if self.not_fullexpand(node):
                    # 添加儿子节点
                    node =  self.expand(node)
                    break
                else:
                    # 若儿子都已在（换言之，儿子均被访问过）则，exploit + explore
                    node = self.bestchild(node,math.sqrt(2))
                    # 给棋盘赋值
                    x ,y = node.position
                    self.board[x][y] = node.player
        return node

    def not_fullexpand(self,node):
        # 这个似乎需要自己定标准 ?  判断儿子节点是否均加入
        return  len(node.potentialChild)

    def expand(self,parentNode):
        # 给父亲节点拓展儿子节点，返回新拓展的儿子节点
        child_position = parentNode.potentialChild.pop(0)
        childNode = MctNode(parent= parentNode, chess_position= child_position, player = 3 - parentNode.player)
        parentNode.child.append(childNode)
        # 给棋盘赋值
        x, y = childNode.position
        self.board[x][y] = childNode.player
        successors = get_successors(board=self.board, player=childNode.player)
        childNode.potentialChild = successors[0:min(len(successors),3)] # 让分枝 <= 某个数
        return childNode

    def defualtpolicy(self,currentPlayer):
        # return 0 / 1 / ... 模拟以计算胜负
        # 若我方有四连，直白赢
        TERMINATE_TIME = 150
        Player = currentPlayer
        newboard = copy.deepcopy(self.board) # 或者递归
        i = 0
        while(i < TERMINATE_TIME):
            # random
            x = random.randint(0, pp.width-1)
            y = random.randint(0, pp.height-1)
            if newboard[x][y] == 0:
                newboard[x][y] = Player
                if if_win(board=newboard,player = Player, x_last = x, y_last = y):
                    return int(currentPlayer == Player)
                Player= 3 - Player
                i += 1
        return 0.05 # 平局 or other evaluate ***************************************************

    def backup(self,node,reward):
        # expand 的 treepolicy 阶段节点开始 反向传播 胜负分数
        while node is not None:
            node.N += 1
            node.value += reward
            node.value_tune += reward**2
            reward = 1 - reward
            if node.position is not None:
                # 赋值回去
                x, y = node.position
                self.board[x][y] = 0
            node = node.parent
        return


def get_successors(board, player):
    # return 当前player的successor
    mykill = get_specialcases(board, player, True,attacker=True)
    if len(mykill['p4']) > 0:
        return mykill['p4']
    opponent = 3 - player
    mythreat = get_specialcases(board, opponent, True,attacker=False)
    if len(mythreat['p4']) > 0:
        return mythreat['p4']
    if len(mykill['l3']) > 0:
        return mykill['l3']
    if len(mythreat['l3']) > 0:
        tmp = mythreat['l3']
        tmp.extend(mykill['s3'])
        return tmp

    record = {}
    for chesspoi in mykill['s3']:
        if chesspoi in record:
            record[chesspoi] += 5
        else:
            record[chesspoi] = 1

    for chesspoi in mykill['l2']:
        if chesspoi in record:
            record[chesspoi] += 5
        else:
            record[chesspoi] = 1

    for chesspoi in mythreat['s3']:
        if chesspoi in record:
            record[chesspoi] += 1
        else:
            record[chesspoi] = 1
    for chesspoi in mythreat['l2']:
        if chesspoi in record:
            record[chesspoi] += 1
        else:
            record[chesspoi] = 1

    if len(record) == 0:
        return nearKsquares(2, board)
    successor = [k for k, v in record.items() if v > 2]
    if len(successor) > 0:
        return successor
    else:
        tmp = nearKsquares(2, board)
        weight = 0.9
        for item in tmp:
            if item not in record:
                record[item] = weight
                weight -= 0.1

        successor = sorted(record.keys(), key=lambda x: record[x], reverse=True)
        return successor
    # 随机
    # 小飞


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

def findChessShape(board, direction, current_position, player, SpecialChess, if_setpoint=True, attacker = True):
    i, j = current_position
    opponent = 3 - player
    deltx, delty = direction
    # 均按一定次序 从左到右 / 从上到下 / 西南-东北 / 西北 - 东南
    # for deltx, delty in [(1, 0), (0, 1), (1, 1), (1, -1)]:
    chessShape = [player]
    x = i
    y = j
    if deltx == 1:
        for x in range(i + 1, min(i + 5, pp.width)):
            y += delty
            if y < 0 or y >= pp.height:
                break
            if board[x][y] == opponent:
                break
            if board[x][y] == 0 and chessShape[-1] == 0:
                chessShape.append(0)
                break
            chessShape.append(board[x][y])
    else:
        for y in range(j + 1, min(j + 5, pp.height)):
            if y < 0 or y >= pp.height:
                break
            if board[x][y] == opponent:
                break
            if board[x][y] == 0 and chessShape[-1] == 0:
                chessShape.append(0)
                break
            chessShape.append(board[x][y])

    x = i - deltx
    y = j - delty
    tmp = sum(chessShape)

    if inBoundary(x,y) and board[x][y] == 0:
        # 左侧无挡
        if tmp == 4 * player:
            if chessShape == [player, player, player, player, 0]:
                SpecialChess['liveFour'] += 1
                if if_setpoint:
                    SpecialChess['p4'].append((x, y))
            else:
                SpecialChess['pushFour'] += 1
                if if_setpoint:
                    if len(chessShape) > 4:
                        SpecialChess['p4'].append((i + deltx * chessShape.index(0), j + delty * chessShape.index(0)))
                    else:
                        SpecialChess['p4'].append((x, y))

        if tmp == 3 * player:
            if len(chessShape) == 5:
                if chessShape == [player, 0, player, 0, player]:
                    SpecialChess['sleepThree'] += 1
                    if if_setpoint:
                        SpecialChess['s3'].append((i + deltx, j + delty))
                        SpecialChess['s3'].append((i + 3 * deltx, j + 3 * delty))
                else:
                    SpecialChess['liveThree'] += 1 # 0 10110 / 0 11010 / 0 11100
                    if if_setpoint:
                        SpecialChess['l3'].append((i + deltx * chessShape.index(0), j + delty * chessShape.index(0)))
                        if not attacker:
                            SpecialChess['l3'].append((x, y))
                            if chessShape[-2] == player:
                                SpecialChess['l3'].append((i + 4 * deltx, j + 4 * delty))

            elif len(chessShape) == 4:
                if chessShape == [player, player, player, 0]:
                    if inBoundary(i - 2 * deltx, j - 2 * delty) and board[i - 2 * deltx][j - 2 * delty] == 0:
                        SpecialChess['liveThree'] += 1
                        if if_setpoint:
                            SpecialChess['l3'].append((i - deltx, j - delty))
                            if not attacker:
                                SpecialChess['l3'].append((i + 3 * deltx, j + 3 * delty))
                                SpecialChess['l3'].append((i - 2 * deltx, j - 2 * delty))
                    else:
                        SpecialChess['sleepThree'] += 1 # 20 11102 ?
                        if if_setpoint:
                            SpecialChess['s3'].append((i - deltx, j - delty))
                            SpecialChess['s3'].append((i + 3 * deltx, j + 3 * delty))
                else:
                    SpecialChess['sleepThree'] += 1 # 011012 /010112
                    if if_setpoint:
                        SpecialChess['s3'].append((i - deltx, j - delty))
                        SpecialChess['s3'].append((i + chessShape.index(0) * deltx, j + chessShape.index(0) * delty))

            elif len(chessShape) == 3 and inBoundary(i - 2 * deltx, j - 2 * delty) and board[i - 2 * deltx][ j - 2 * delty] == 0 :
                SpecialChess['sleepThree'] += 1
                if if_setpoint:
                    SpecialChess['s3'].append((i - deltx, j - delty))

        if tmp == 2 * player:
            if len(chessShape) == 4:
                if chessShape == [player, player, 0, 0]:
                    if inBoundary(i + 4 * deltx, j + 4 * delty) and board[i + 4 * deltx][j + 4 * delty] == player:
                        SpecialChess['sleepThree'] += 1 # 011001?
                        if if_setpoint:
                            SpecialChess['s3'].append((i + 2 * deltx, j + 2 * delty))
                            SpecialChess['s3'].append((i + 3 * deltx, j + 3 * delty))
                    elif inBoundary(i - 2 * deltx, j - 2 * delty) and board[i - 2 * deltx][j - 2 * delty] == 0:
                        SpecialChess['liveTwo'] += 1 # 0011002
                        if if_setpoint:
                            SpecialChess['l2'].append((i - deltx, j - delty))
                            if inBoundary(i + 4 * deltx, j + 4 * delty) and board[i + 4 * deltx][j + 4 * delty] == 0:
                                SpecialChess['l2'].append((i + 2*deltx, j + 2*delty))
                            if inBoundary(i - 3 * deltx, j - 3 * delty) and board[i - 3 * deltx][j - 3 * delty] == 0:
                                SpecialChess['l2'].append((i - 2 * deltx, j - 2 * delty))
                    else:
                        SpecialChess['sleepTwo'] += 1  # some question

                else:
                    SpecialChess['sleepTwo'] += 1

            elif len(chessShape) == 5: # [1,0,1,0,0]
                SpecialChess['liveTwo'] += 1
                if if_setpoint:
                    SpecialChess['l2'].append((i + deltx, j + delty))
                    SpecialChess['l2'].append((i + 3*deltx, j + 3*delty))


        if tmp == player and len(chessShape) == 3:
            # maybe live three
            # maybe live Two
            x = i + 2 * deltx
            y = j + 2 * delty
            newshape = []
            for _ in range(2):
                x += deltx
                y += delty
                if inBoundary(x,y) and board[x][y] != opponent:
                    newshape.append(board[x][y])
                else:
                    break
            if newshape == [player, player]:
                SpecialChess['sleepThree'] += 1
                if if_setpoint:
                    SpecialChess['s3'].append((i + deltx, j + delty))
                    SpecialChess['s3'].append((i + 2 * deltx, j + 2 * delty))
            elif newshape == [player, 0]:
                SpecialChess['liveTwo'] += 1
                if if_setpoint:
                    SpecialChess['l2'].append((i + deltx, j + delty))
                    SpecialChess['l2'].append((i + 2 * deltx, j + 2 * delty))

            elif newshape == [player]:
                SpecialChess['sleepTwo'] += 1

    elif not inBoundary(x, y) or board[x][y] == opponent:
        if tmp == 4 * player:  # ??
            if len(chessShape) == 5:
                SpecialChess['pushFour'] += 1
                if if_setpoint:
                    SpecialChess['p4'].append((i + chessShape.index(0) * deltx, j + chessShape.index(0) * delty))
        if tmp == 3 * player:
            if len(chessShape) == 5:
                SpecialChess['sleepThree'] += 1
                if if_setpoint:
                    SpecialChess['s3'].append((i + chessShape.index(0) * deltx, j + chessShape.index(0) * delty))
                    if chessShape == [player, 0, player, 0, player]:
                        SpecialChess['s3'].append((i + 3 * deltx, j + 3 * delty))
                    else:
                        SpecialChess['s3'].append((i + 4 * deltx, j + 4 * delty))

def get_specialcases(board, player, if_setpoint=False,attacker = False):
    '''
    :param board:
    :param player:
    :param if_setpoint:
    :param attacker: 是否是攻击模式 // 譬如 活三的多种防御法 与 进攻冲活四的矛盾
    :return:
    '''
    SpecialChess = {'renju': 0, 'liveFour': 0, 'pushFour': 0,
                    'liveThree': 0, 'sleepThree': 0, 'liveTwo': 0, 'sleepTwo': 0,
                    'p4': [], 'l3': [], 's3': [], 'l2': []}
    for i in range(pp.width):
        for j in range(pp.height):
            if board[i][j] != player:
                continue
            for direction in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                findChessShape(board=board, direction=direction, player=player, current_position=(i, j),
                               SpecialChess=SpecialChess, if_setpoint=if_setpoint,attacker=attacker)
    return SpecialChess

# MCTS METHOD ***************************************************************** END

def Mybrain_turn():
    if pp.terminateAI:
        return
    newboard = []  # shrinkage
    for rowindex in range(pp.width):
        tmp = board[rowindex]
        newboard.append(copy.deepcopy(tmp[0:pp.height]))

    mct = MCTS(board=newboard, player=1)
    bestnode = mct.UCTsearch(IterMAX=2000)
    x, y = bestnode.position
    pp.do_mymove(x, y)


def brain_turn():
    if pp.terminateAI:
        return

    newboard = []  # shrinkage
    for rowindex in range(pp.width):
        tmp = board[rowindex]
        newboard.append(copy.deepcopy(tmp[0:pp.height]))

    logDebug("some message 1: for MC:")
    mct = MCTS(board=newboard, player=1)
    bestnode = mct.UCTsearch(IterMAX=100)
    x, y = bestnode.position
    pp.do_mymove(x, y)

    pp.do_mymove(x, y)




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
# A possible way how to debug brains.
# To test it, just "uncomment" it (delete enclosing """)
######################################################################
"""
# define a file for logging ...
DEBUG_LOGFILE = "/tmp/pbrain-pyrandom.log"
# ...and clear it initially
with open(DEBUG_LOGFILE,"w") as f:
	pass

# define a function for writing messages to the file
def logDebug(msg):
	with open(DEBUG_LOGFILE,"a") as f:
		f.write(msg+"\n")
		f.flush()

# define a function to get exception traceback
def logTraceBack():
	import traceback
	with open(DEBUG_LOGFILE,"a") as f:
		traceback.print_exc(file=f)
		f.flush()
	raise

# use logDebug wherever
# use try-except (with logTraceBack in except branch) to get exception info
# an example of problematic function
def brain_turn():
	logDebug("some message 1")
	try:
		logDebug("some message 2")
		1. / 0. # some code raising an exception
		logDebug("some message 3") # not logged, as it is after error
	except:
		logTraceBack()
"""
######################################################################

# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = Mybrain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about
if DEBUG_EVAL:
    pp.brain_eval = brain_eval


def main():
    pp.main()


if __name__ == "__main__":
    main()
