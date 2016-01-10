
import random

DEFAULT_CAPACITY = 1000

class ReplayMemory(object):
    def __init__(self, capacity=DEFAULT_CAPACITY):
        self.memory = {}
        self.first_index = -1
        self.last_index = -1
        self.capacity = capacity

    def store(self, sars_tuple):
        if self.first_index == -1:
            self.first_index = 0
        self.last_index += 1
        self.memory[self.last_index] = sars_tuple   
        if (self.last_index + 1 - self.first_index) > self.capacity:
            self.discardSample()

    def isFull(self):
        return self.last_index + 1 - self.first_index >= self.capacity

    def discardSample(self):
        rand_index = random.randint(self.first_index, self.last_index)
        first_tuple = self.memory[self.first_index]
        del self.memory[rand_index]
        if rand_index != self.first_index:
            del self.memory[self.first_index]
            self.memory[rand_index] = first_tuple
        self.first_index += 1

    def sample(self):
        if self.first_index == -1:
            return
        rand_sample_index = random.randint(self.first_index, self.last_index)
        return self.memory[rand_sample_index]
