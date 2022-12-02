# Student agent: Add your own agent here
import time
from copy import deepcopy

import numpy as np

from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return
        dim = len(chess_board[0])
        M = Monte(chess_board, my_pos, adv_pos, max_step)
        steps = M.get_all_steps(chess_board, my_pos, adv_pos, max_step, dim)
        #print(steps)
        my_pos, direction = M.MTCL(chess_board, my_pos, adv_pos, max_step, steps)

        # return my_pos, self.dir_map["u"]
        return my_pos, direction


class Monte:
    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        # self.board_size = board_dim = len(chess_board[0])
        self.count = 0
        self.max_step = max_step
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.setup = True
        self.wins = {}
        self.plays = {}
        self.num = 0
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def MTCL(self, chess_board, my_pos, adv_pos, max_step, steps):
        try:
            if self.__class__.MTCL.called:
                # do what you have to do with the method
                print("normal execution")
                count=1
        except AttributeError:
            # do what you have to do with the first call
            print("first call")
            count = 0
            self.__class__.MTCL.called = True

        #count = 0
        if count==0:
            sec = 20
        else:
            sec = 1.95
        dict = {}
        start_time = time.time()
        #sec = 1.95

        # while True:
        #     cur = time.time()
        #     end = cur-start_time
        #     if(end>sec):
        #         print("break")
        #         break
        print("total:")
        print(len(steps))
        m = 0
        for move in steps:
                #print(move)
            board_copy = deepcopy(chess_board)
            r, c, dir = move
            self.set_barrier(board_copy, r, c, dir)

            S = self.run_simulation(board_copy, (r, c), adv_pos, max_step, count)

            move = ((r, c), dir)
            dict[move] = S
            cur = time.time()
            # if count == 0:
            #     end = cur-start_time
            # else:
            #     end = cur - (start_time+20+2*(count-1))
            end = cur - start_time
            if end > sec:
                #print("break")
                break
            m+=1

        #choose = dict[max(dict, key=dict.get)]
        print("see:")
        print(m)
        me_time = time.time()
        choose = max(dict, key=lambda key: dict[key])
        posr, dirr = choose
        end_time = time.time()
        print(end_time-start_time)
        #count+=1
        return posr, dirr

    def simulation(self, chess_board, my_pos, adv_pos, turn, max_step):

        board = deepcopy(self.chess_board)

        board_size = len(board[0])
        res = self.check_endgame(board, my_pos, adv_pos, max_step, board_size)

        if res[1]==-25:
            return -150
        elif res[1]==100:
            return 1000000
        elif res[1]==-10000:
            return -1000000



        while not res[0]:

            if turn == 0:
                random = self.random_moves(board, my_pos, adv_pos, max_step)
                my_pos = random[0]
                my_dir = random[1]
                self.set_barrier(board, my_pos[0], my_pos[1], my_dir)
                #print("here")
                turn = 1
            elif turn == 1:
                random = self.random_moves(board, adv_pos, my_pos, max_step)
                adv_pos = random[0]
                adv_dir = random[1]
                self.set_barrier(board, adv_pos[0], adv_pos[1], adv_dir)
                #print("there")
                turn = 0
            res = self.check_endgame(board, my_pos, adv_pos, max_step, board_size)
            #print(res[0])
            #print(my_pos)
            #print(adv_pos)
        return res[1]

    def run_simulation(self, chess_board, my_pos, adv_pos, max_step,count):
        turn = 0
        sum = 0
        if count==0:
            numSims = 20
        else:
            numSims = 4

        for i in range(numSims):
            board1 = deepcopy(chess_board)
            sum += self.simulation(board1, my_pos, adv_pos, turn, max_step)
        return sum

    def get_all_steps(self, chess_board, my_pos, adv_pos, max_step, dim):
        all_valids = []
        for r in range(max(0, my_pos[0] - max_step), min(dim, my_pos[0] + max_step)):
            for c in range(max(0, my_pos[1] - max_step), min(dim, my_pos[1] + max_step)):
                if (abs(my_pos[0] - r) + abs(my_pos[1] - c)) in range(0, max_step + 1):
                    for key in list(self.dir_map):
                        dire = self.dir_map[key]
                        if self.check_valid_step(chess_board, my_pos, (r, c), dire, adv_pos, max_step):
                            all_valids.append((r, c, dire))
        return all_valids

    def set_barrier(self, chess_board, r, c, dir):
        chess_board[int(r), int(c), int(dir)] = True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def check_valid_step(self, chess_board, start_pos, end_pos, barrier_dir, adv_pos, max_step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        # adv_pos = self.p0_pos if self.turn else self.p1_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            # print(cur_pos)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue

                # next_pos = cur_pos + move
                next_pos = cur_pos[0] + move[0], cur_pos[1] + move[1]
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def random_moves(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)
        #print(chess_board)

        # Random Walkpython simulator.py --player_1 student_agent --player_2 random_agent
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    def check_endgame(self, chess_board, my_pos, adv_pos, max_step, board_size):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            #print(1)
            return False, 0
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
            return True, -25

        if (p0_score - p1_score) < 0:
            return True, -10000
        else:
            return True, 100
