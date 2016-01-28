import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import unittest
import numpy as np

import mdps

class TestMazeMDP(unittest.TestCase):

    """ runs_into_wall tests """
    def test_leave_maze_negative_x(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=1)
        state = (0,0)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_leave_maze_positive_x(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=1)
        state = (4,0)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_leave_maze_negative_y(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=1)
        state = (0,0)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_leave_maze_positive_y(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=1)
        state = (0,4)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_leave_maze_negative_x_false(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=1)
        state = (1,0)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_leave_maze_positive_x_false(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=1)
        state = (3,0)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_leave_maze_negative_y_false(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=1)
        state = (0,1)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_leave_maze_positive_y_false(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=1)
        state = (0,3)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_x_right_to_left(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (4,0)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_wall_cross_x_left_to_right(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (5,0)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_wall_cross_y_down_to_up(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (0,4)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_wall_cross_y_up_to_down(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (0,5)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_wall_cross_x_right_to_left_false(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (3,0)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_x_left_to_right_false(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (6,0)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_y_down_to_up_false(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (0,3)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_y_up_to_down_false(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (0,6)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_x_right_to_left(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (4,2)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_x_left_to_right(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (5,2)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_y_up(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (2,4)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_y_down(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=2)
        state = (2,5)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    """ runs_into_wall tests on larger mazes """
    def test_leave_maze_negative_x_larger(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=5)
        state = (0,0)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_leave_maze_negative_y_larger(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=5)
        state = (0,0)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_leave_maze_negative_x_false_larger(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=5)
        state = (1,0)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_leave_maze_positive_x_false_larger(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=5)
        state = (3,0)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_leave_maze_negative_y_false_larger(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=5)
        state = (0,1)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_leave_maze_positive_y_false_larger(self):
        mdp = mdps.MazeMDP(room_size=5, num_rooms=5)
        state = (0,3)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_x_right_to_left_larger(self):
        mdp = mdps.MazeMDP(room_size=3, num_rooms=2)
        state = (2,4)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_x_left_to_right_larger(self):
        mdp = mdps.MazeMDP(room_size=3, num_rooms=2)
        state = (3,4)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_y_up_larger(self):
        mdp = mdps.MazeMDP(room_size=3, num_rooms=2)
        state = (4,2)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_y_down_larger(self):
        mdp = mdps.MazeMDP(room_size=3, num_rooms=2)
        state = (4,3)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    """ runs_into_wall tests on different room sizes """

    def test_wall_cross_x_right_to_left_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (6,0)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_wall_cross_x_left_to_right_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (7,0)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_wall_cross_y_down_to_up_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (0,6)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_wall_cross_y_up_to_down_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (0,7)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = True
        self.assertEquals(actual, expected)

    def test_wall_cross_x_right_to_left_false_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (3,0)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_x_left_to_right_false_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (6,0)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_y_down_to_up_false_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (0,3)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_y_up_to_down_false_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (0,6)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_x_right_to_left_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (6,3)
        action = (1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_x_left_to_right_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (7,3)
        action = (-1,0)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_y_up_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (3,6)
        action = (0,1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

    def test_wall_cross_through_doorway_y_down_larger_room_size(self):
        mdp = mdps.MazeMDP(room_size=7, num_rooms=2)
        state = (3,7)
        action = (0,-1)
        actual = mdp.runs_into_wall(state, action)
        expected = False
        self.assertEquals(actual, expected)

if __name__ == '__main__':
    # this runs all tests in file if executed directly (e.g., python test_nnet.py)
    # run a single test by specifying the name (e.g., python test_nnet.py 
    # TestNNet.test_loss_updates_one_layer_positive_relu)
    unittest.main()
