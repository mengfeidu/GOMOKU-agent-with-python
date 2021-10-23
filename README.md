# GOMOKU-agent-with-python
This project is the final project of DATA130008.01 Introduction to Artificial Intelligence(2020)

## Introduction
* The project requirements can be seen in the file **final_project.pdf**
* The main codes can be seen in the files **MCTS_Gomoku.py** ,**Final_Gomoku.py** and **Greedy.py** which are Gomoku agents implemented with MCTS , alpha-beta pruning and greedy algorithm respectively.
* The details of this project can be seen in the file **report.pdf**
* The agents are **pbrain-pyfinal.exe** and **pbrain-mcts.exe** , you can download **piskvork** to use them play with other agents and write your own agents.

## Performance of Our Agents
###  alpha-beta pruning
| Agent Name        | Win   |  Loss  |
| --------   | -----:  | :----:  |
| PUREROCKY     | 11 |   1     |
| PELA17       |   1   |   11  |
| NOESIS        |    6   |  6  |
| ZETOR17        |    1   |  11  |
| FIVEROW        |   10   |  2  |
| SPARKLE        |    4   |  8  |
| VALKYRIE        |    12   |  0  |
| PISQ7        |    6   |  6  |
| EULRING        |    2   |  10  |
| YIXIN        |    1   |  11  |
| WINE        |    0   |  12  |
| MUSHROOM       |    12   |  0  |
| Total | Final Elo Rating   | 1395.0  |
###  MCTS
| Agent Name        | Win   |  Loss  |
| --------   | -----:  | :----:  |
| PUREROCKY     | 4 |   8     |
| PELA17       |   0   |   12  |
| NOESIS        |    1   |  11  |
| ZETOR17        |    0   |  12  |
| FIVEROW        |   6   |  6  |
| SPARKLE        |    0   |  12  |
| VALKYRIE        |    3   |  9  |
| PISQ7        |    2   |  10  |
| EULRING        |    0  |  12  |
| YIXIN        |    0   |  12  |
| WINE        |    0   |  12  |
| MUSHROOM       |    7  |  5  |
| Total | Final Elo Rating   | 942.0  |
###  Greedy
| Agent Name        | Win   |  Loss  |
| --------   | -----:  | :----:  |
| PUREROCKY     | 9 |   3     |
| PELA17       |   0   |   12  |
| NOESIS        |    0   |  12  |
| ZETOR17        |    0   |  12  |
| FIVEROW        |  9  | 3  |
| SPARKLE        |    0   |  12  |
| VALKYRIE        |    5   | 7  |
| PISQ7        |    3   |  9  |
| EULRING        |    1  |  11  |
| YIXIN        |    0   |  12  |
| WINE        |    0   |  12  |
| MUSHROOM       |    11  |  1  |
| Total | Final Elo Rating   | 1111.0  |
