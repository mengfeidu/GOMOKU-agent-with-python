import random
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
import math
import copy
import time

pp.infotext = 'name="pbrain-pyrandom", author="Jan Stransky", version="1.0", country="Czech Republic", www="https://github.com/stranskyjan/pbrain-pyrandom"'

MAX_BOARD = 100
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]

THRESHOLD = 3
inBoundary = lambda x, y: (x >= 0 and y >= 0 and x < pp.width and y < pp.height)

# kill chess initialization
recordWin = {}


def get_zobrist(hashTable):
    for i in range(pp.width):
        for j in range(pp.height):
            hashTable[(i, j)] = random.randint(0, 1e30)
    return


AlreadyKill = False  # 检测是否已杀

hashTableForP1 = {}  # Zobrist for player 1
get_zobrist(hashTableForP1)

hashTableForP2 = {}
get_zobrist(hashTableForP2)


def chess2int(board):
    # 将棋盘转化成一个整数
    chessTable = 0
    for x in range(pp.width):
        for y in range(pp.height):
            if board[x][y] == 0:
                continue
            elif board[x][y] == 1:
                chessTable ^= hashTableForP1[(x, y)]
            else:
                chessTable ^= hashTableForP2[(x, y)]
    return chessTable


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
        x = random.randint(0, pp.width)
        y = random.randint(0, pp.height)
        i += 1
        if pp.terminateAI:
            return
        if isFree(x, y):
            break
    if i > 1:
        pp.pipeOut("DEBUG {} coordinates didn't hit an empty field".format(i))
    pp.do_mymove(x, y)


def Mybrain_turn():
    begin = time.time()
    newboard = []  # shrinkage
    for rowindex in range(pp.width):
        tmp = board[rowindex]
        newboard.append(copy.deepcopy(tmp[0:pp.height]))

    # 杀棋的总体效果不佳 ...... 特例时 嗖的一下 切实相当爽
    # chessnum = chess2int(newboard)
    # if chessnum in recordWin:
    #     x, y = recordWin[chessnum]
    #     if isFree(x, y):
    #         return (x, y)
    #
    # fkt = FKTexplore(board=board, threshold=10)
    # fkt.fktSolution()
    # if fkt.winChoicePosi is not None:
    #     x, y = fkt.winChoicePosi
    #     pp.do_mymove(x, y)
    #     return

    successor = get_successors(newboard, player=1)
    player = 1  # 假设是我方轮次
    best_position = None
    alpha0 = -20000  # 必输分数
    beta0 = 10000  # 必赢分数
    for position in successor:
        if time.time() - begin > 14.99:
            break
        x1, y1 = position
        newboard[x1][y1] = player
        if len(successor) > 1:
            tmp_value = value(depth=0, player=1, alpha=alpha0, beta=beta0, board=newboard, position=position)
        else:
            best_position = position
            break
        newboard[x1][y1] = 0  # 还原
        if tmp_value >= 10000:  # Win
            best_position = position
            break
        elif tmp_value > alpha0:
            best_position = position
            alpha0 = tmp_value

    x, y = best_position
    pp.do_mymove(x, y)


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


def findChessShape(board, direction, current_position, player, SpecialChess, if_setpoint=True):
    i, j = current_position
    opponent = 3 - player
    deltx, delty = direction
    # 均按一定次序 从左到右 / 从上到下 / 西南-东北 / 西北 - 东南
    # for deltx, delty in [(1, 0), (0, 1), (1, 1), (1, -1)]:
    # 大一统模型，搞起
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

    if inBoundary(x, y) and board[x][y] == 0:
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
                    SpecialChess['liveThree'] += 1
                    if if_setpoint:
                        SpecialChess['l3'].append((x, y))
                        SpecialChess['l3'].append((i + deltx * chessShape.index(0), j + delty * chessShape.index(0)))
                        if chessShape[-2] == player:
                            SpecialChess['l3'].append((i + 4 * deltx, j + 4 * delty))

            elif len(chessShape) == 4:
                if chessShape == [player, player, player, 0]:
                    if inBoundary(i - 2 * deltx, j - 2 * delty) and board[i - 2 * deltx][j - 2 * delty] == 0:
                        SpecialChess['liveThree'] += 1
                        if if_setpoint:
                            SpecialChess['l3'].append((i - deltx, j - delty))
                            SpecialChess['l3'].append((i + 3 * deltx, j + 3 * delty))
                    else:
                        SpecialChess['sleepThree'] += 1
                        if if_setpoint:
                            SpecialChess['s3'].append((i - deltx, j - delty))
                            SpecialChess['s3'].append((i + 3 * deltx, j + 3 * delty))
                else:
                    SpecialChess['sleepThree'] += 1
                    if if_setpoint:
                        SpecialChess['s3'].append((i - deltx, j - delty))
                        SpecialChess['s3'].append((i + chessShape.index(0) * deltx, j + chessShape.index(0) * delty))

            elif len(chessShape) == 3 and inBoundary(i - 2 * deltx, j - 2 * delty) and board[i - 2 * deltx][
                j - 2 * delty] == 0:
                SpecialChess['sleepThree'] += 1
                if if_setpoint:
                    SpecialChess['s3'].append((i - deltx, j - delty))

        if tmp == 2 * player:
            if len(chessShape) == 4:
                if chessShape == [player, player, 0, 0]:
                    if inBoundary(i + 4 * deltx, j + 4 * delty) and board[i + 4 * deltx][j + 4 * delty] == player:
                        SpecialChess['sleepThree'] += 1
                        if if_setpoint:
                            SpecialChess['s3'].append((i + 2 * deltx, j + 2 * delty))
                            SpecialChess['s3'].append((i + 3 * deltx, j + 3 * delty))
                    elif inBoundary(i - 2 * deltx, j - 2 * delty) and board[i - 2 * deltx][j - 2 * delty] == 0:
                        SpecialChess['liveTwo'] += 1
                        if if_setpoint:
                            SpecialChess['l2'].append((i - deltx, j - delty))
                            if inBoundary(i + 4 * deltx, j + 4 * delty) and board[i + 4 * deltx][j + 4 * delty]:
                                SpecialChess['l2'].append((i + 2 * deltx, j + 2 * delty))
                            if inBoundary(i - 3 * deltx, j - 3 * delty) and board[i - 3 * deltx][j - 3 * delty] == 0:
                                SpecialChess['l2'].append((i - 2 * deltx, j - 2 * delty))
                    else:
                        SpecialChess['sleepTwo'] += 1  # some question

                else:
                    SpecialChess['sleepTwo'] += 1

            elif len(chessShape) == 5:  # [1,0,1,0,0]
                SpecialChess['liveTwo'] += 1
                if if_setpoint:
                    SpecialChess['l2'].append((i + deltx, j + delty))
                    SpecialChess['l2'].append((i + 3 * deltx, j + 3 * delty))

        if tmp == player and len(chessShape) == 3:
            # maybe live three
            # maybe live Two
            x = i + 2 * deltx
            y = j + 2 * delty
            newshape = []
            for _ in range(2):
                x += deltx
                y += delty
                if inBoundary(x, y) and board[x][y] != opponent:
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


def get_specialcases(board, player, if_setpoint=False):
    SpecialChess = {'renju': 0, 'liveFour': 0, 'pushFour': 0,
                    'liveThree': 0, 'sleepThree': 0, 'liveTwo': 0, 'sleepTwo': 0,
                    'p4': [], 'l3': [], 's3': [], 'l2': []}
    for i in range(pp.width):
        for j in range(pp.height):
            if board[i][j] != player:
                continue
            for direction in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                findChessShape(board=board, direction=direction, player=player, current_position=(i, j),
                               SpecialChess=SpecialChess, if_setpoint=if_setpoint)
    return SpecialChess


class FKTexplore:
    '''
    fast kill test
    趁Ta病,要Ta命, 一鼓作气再而衰三而竭
    '''

    def __init__(self, board, threshold, player=1):
        self.board = board
        self.threshold = threshold  # fast kill 搜索层数限制
        self.player = player  # 查一波我方杀棋

        self.threatForopponent = []  # 对方可能应对措施 e.g. 冲四 / 堵我方一手
        self.lastPosi = None

        self.winChoicePosi = None  # only win will be set
        # zobrist 唯一标识
        self.chessnum = chess2int(self.board)
        # 记录已下过棋形
        self.endfkt = {}

    def fktvalue(self, depth, chesshape):
        # 不成功便成仁
        x, y = self.lastPosi

        if if_win(board=self.board, player=self.player, x_last=x, y_last=y):
            if self.player == 1:
                return 1
            else:
                return 0

        if depth < self.threshold:
            depth += 1
        else:
            return 0

        # continue to alpha - beta

        if self.player == 1:
            # player 2's turn
            self.player = 3 - self.player
            return self.fktmin(depth=depth, chesshape=chesshape)
        else:
            # player 1's turn
            self.player = 3 - self.player
            return self.fktmax(depth=depth, chesshape=chesshape)

    def fktmin(self, depth, chesshape):
        v = 1
        successors = self.fktSuccessor()
        if len(successors) == 0:
            return 0
        for new_posi in successors:
            x, y = new_posi
            self.board[x][y] = self.player  # 2
            self.lastPosi = (x, y)

            chesshape ^= hashTableForP2[(x, y)]

            if chesshape in self.endfkt:
                v = self.endfkt[chesshape]
            else:
                v = min(v, self.fktvalue(depth, chesshape))
                self.endfkt[chesshape] = v

            # traceback
            chesshape ^= hashTableForP2[(x, y)]
            self.board[x][y] = 0
            if v == 0:  # player 1 没赢
                global AlreadyKill
                AlreadyKill = False
                return v
        return v

    def fktmax(self, depth, chesshape):
        # 该步为player1 ，我方落子
        v = 0
        successors = self.fktSuccessor()
        if len(successors) == 0:
            return 0  # 无棋可杀
        for new_posi in successors:  # player 1 的可能走法
            x, y = new_posi
            self.board[x][y] = self.player
            self.lastPosi = (x, y)
            chesshape ^= hashTableForP1[(x, y)]
            if chesshape in self.endfkt:
                v = self.endfkt[chesshape]
            else:
                v = max(v, self.fktvalue(depth=depth, chesshape=chesshape))
                self.endfkt[chesshape] = v

            # traceback
            chesshape ^= hashTableForP1[(x, y)]
            self.board[x][y] = 0
            if v == 1:
                # 杀棋
                global AlreadyKill
                AlreadyKill = True
                recordWin[chesshape] = (x, y)
                return v
        return v

    def fktSuccessor(self):
        # 核心之核心 但拉倒了 rewrite
        if self.player == 1:
            # 找到 (活三、活四、冲四）威胁 / 阻止对手的 冲四 威胁
            mykill = get_specialcases(self.board, self.player, if_setpoint=True, attacker=True)
            mythreat = get_specialcases(self.board, 3 - self.player, if_setpoint=True, attacker=False)
            if len(mykill['p4']) > 0:
                return mykill['p4']
            if len(mythreat['p4']) > 0:
                return mythreat['p4']  # 阻止冲四
            if len(mykill['l3']) > 0:
                return mykill['l3']  # 成活四
            if len(mythreat['l3']) > 0:
                return mykill['s3']
            else:
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
                successor = sorted(record.keys(), key=lambda x: record[x], reverse=True)
                return successor
        else:
            # 找到 (活四、冲四) 威胁 / 阻止对手的 冲四、活三威胁
            mykill = get_specialcases(self.board, self.player, if_setpoint=True, attacker=True)
            mythreat = get_specialcases(self.board, 3 - self.player, if_setpoint=True, attacker=False)
            if len(mykill['p4']) > 0:
                return mykill['p4']
            if len(mythreat['p4']) > 0:
                return mythreat['p4']  # 阻止冲四
            if len(mykill['l3']) > 0:
                return mykill['l3']  # 成活四
            return mythreat['l3']

    def fktSolution(self):
        '''
        算出是否有杀棋
        :return:
        '''

        player = self.player
        chesshape = self.chessnum
        successors = self.fktSuccessor()  # ???

        for posi in successors:
            x, y = posi
            self.board[x][y] = player
            self.lastPosi = posi
            chesshape ^= hashTableForP1[(x, y)]

            v = self.fktvalue(depth=0, chesshape=chesshape)
            self.endfkt[chesshape] = v
            # traceback
            chesshape ^= hashTableForP1[(x, y)]
            self.board[x][y] = 0
            if v == 1:
                # 杀棋成功
                global AlreadyKill
                AlreadyKill = True
                tmp = chess2int(self.board)
                recordWin[tmp] = (x, y)
                # 只起此作用
                self.winChoicePosi = (x, y)
                return
        return


def get_successors(board, player):
    # return 当前player的successor
    mykill = get_specialcases(board, player, True)
    if len(mykill['p4']) > 0:
        return mykill['p4']
    opponent = 3 - player
    mythreat = get_specialcases(board, opponent, True)
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


def evaluate(board, player):
    score = {'renju': 0, 'liveFour': 10000, 'pushFour': 2,
             'liveThree': 2, 'sleepThree': 1.5, 'liveTwo': 1, 'sleepTwo': 0.2}
    # score = {'renju': 10000,
    #               'L4': 1000,
    #               'S4': 4,
    #               'L3': 4,
    #               'S3': 2,
    #               'L2': 2,
    #               'S2': 1,
    #               'D4': -2,
    #               'D3': -2,
    #               'D2': -2}
    if player == 1:
        # 我方刚下完最后一步，相同棋形，敌方占优,敌方有四连直接赢
        chessSituation = get_specialcases(board, player=1)
        chessSituation_opponent = get_specialcases(board, player=2)
        allscore = 0
        if chessSituation_opponent['liveFour'] + chessSituation_opponent['pushFour'] >= 1:
            return -10000  # 我方直接gg
        elif chessSituation_opponent['liveThree'] >= 1 and (
                chessSituation['pushFour'] + chessSituation['liveFour'] == 0):
            return -10000  # 我方同样直接gg
        if chessSituation['liveThree'] > 1:
            score['liveThree'] *= 10
        if chessSituation['pushFour'] > 1:
            score['pushFour'] *= 10

        for item in score:
            extra = 0.1 * score[item] * int(
                (chessSituation[item] > 0) and (chessSituation[item] == chessSituation_opponent[item]))
            allscore += score[item] * (chessSituation[item] - chessSituation_opponent[item]) - extra
        return allscore
    else:
        # 此步是对方下完，我方先走，相同棋形，我方占优，我方四连直接赢
        chessSituation = get_specialcases(board, player=1)
        chessSituation_opponent = get_specialcases(board, player=2)
        allscore = 0
        if chessSituation['liveFour'] + chessSituation['pushFour'] >= 1:
            return +10000  # 我方直接win
        elif chessSituation['liveThree'] >= 1 and (
                chessSituation_opponent['pushFour'] + chessSituation_opponent['liveFour'] == 0):
            return +10000  # 我方同样直接win

        if chessSituation_opponent['liveThree'] > 1:
            score['liveThree'] *= 10
        if chessSituation_opponent['pushFour'] > 1:
            score['pushFour'] *= 10

        for item in score:
            extra = 0.1 * score[item] * int(
                (chessSituation[item] > 0) and (chessSituation[item] == chessSituation_opponent[item]))
            allscore += score[item] * (chessSituation[item] - chessSituation_opponent[item]) + extra
        return allscore


def value(depth, player, board, position, alpha, beta):
    x, y = position

    if if_win(board=board, player=player, x_last=x, y_last=y):
        if player == 1:
            return 10000
        else:
            return -10000

    if depth < THRESHOLD:
        depth += 1

    else:
        return evaluate(board, player=player)
        # test

    if player == 1:
        # player 2's turn
        return min_value(depth=depth, player=2, board=board, position=position, alpha=alpha, beta=beta)
    else:
        # player 1's turn
        return max_value(depth=depth, player=1, board=board, position=position, alpha=alpha, beta=beta)


def max_value(depth, player, board, position, alpha, beta):
    # 该步为player1 ，我方落子
    v = -math.inf
    successors = get_successors(board, player=player)
    for new_posi in successors:  # player 1 的可能走法
        x, y = new_posi
        board[x][y] = player
        v = max(v, value(depth=depth, player=1, board=board, position=new_posi, alpha=alpha, beta=beta))
        # traceback
        board[x][y] = 0
        alpha = max(alpha, v)
        if alpha >= beta:
            return v
    return v


def min_value(depth, player, board, position, alpha, beta):
    # 该步为player2 ，对方轮次
    v = math.inf
    successors = get_successors(board, player=player)
    for new_posi in successors:
        x, y = new_posi

        board[x][y] = player
        v = min(v, value(depth, 2, board, new_posi, alpha, beta))
        # traceback
        board[x][y] = 0
        beta = min(beta, v)
        if alpha >= beta:
            return v
    return v


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

# define a file for logging ...
# DEBUG_LOGFILE = "F:/Gomokuoutcome.log"
# # ...and clear it initially
# with open(DEBUG_LOGFILE, "w") as f:
#     pass


# define a function for writing messages to the file
# def logDebug(msg):
#     with open(DEBUG_LOGFILE, "a") as f:
#         f.write(msg + "\n")
#         f.flush()


# define a function to get exception traceback
# def logTraceBack():
#     import traceback
#     with open(DEBUG_LOGFILE, "a") as f:
#         traceback.print_exc(file=f)
#         f.flush()
#     raise


# use logDebug wherever
# use try-except (with logTraceBack in except branch) to get exception info
# an example of problematic function
# def brain_turn():
#     logDebug("some message 1")
#     try:
#         logDebug("some message 2")
#         1. / 0.  # some code raising an exception
#         logDebug("some message 3")  # not logged, as it is after error
#     except:
#         logTraceBack()


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
