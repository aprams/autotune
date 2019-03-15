import random
import numpy as np
from abc import ABC, abstractmethod


class Param(ABC):
    @abstractmethod
    def __init__(self, space=None, projection_fn=None, name="Param"):
        self.space = space
        if projection_fn is not None:
            self.projection_fn = projection_fn
        else:
            self.projection_fn = lambda x: x
        self.name = name

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def transform_raw_sample(self, raw_sample):
        pass

    @abstractmethod
    def raw_sample(self):
        pass

    @abstractmethod
    def create_generator(self):
        pass


class Integer(Param):
    def __init__(self, space=None, projection_fn=None, name="IntegerParam"):
        if space is None:
            space = [0, 1]
        super(Integer, self).__init__(space, projection_fn, name)


    def sample(self):
        return self.transform_raw_sample(self.raw_sample())

    def transform_raw_sample(self, raw_sample):
        return self.projection_fn(self.space[int(raw_sample)])

    def raw_sample(self):
        rand_item = random.randint(0, len(self.space) - 1)
        return rand_item

    def create_generator(self):
        def generator():
            for i in self.space:
                yield self.projection_fn(i)
        return generator


class Real(Param):
    def __init__(self, space=None, projection_fn=None, name="RealParam", n_points_to_sample=50):
        if space is None:
            space = [0, 1]
        if len(space) != 2:
            raise Exception("RealParam expects list of length two with: [lower_bound_inclusive, upper_bound_inclusive]")
        self.lower_bound = space[0]
        self.upper_bound = space[1]
        self.n_points_to_sample = n_points_to_sample
        super(Real, self).__init__(space, projection_fn, name)

    def sample(self):
        return self.transform_raw_sample(self.raw_sample())

    def transform_raw_sample(self, raw_sample):
        return self.projection_fn(raw_sample)

    def raw_sample(self):
        random_sample = random.uniform(self.lower_bound, self.upper_bound)
        return random_sample

    def create_generator(self):
        def generator():
            sampled_xs = np.linspace(self.lower_bound, self.upper_bound, self.n_points_to_sample)
            for i in sampled_xs:
                yield self.projection_fn(i)
        return generator


class Bool(Integer):
    def __init__(self, space=None, projection_fn=None, name="BoolParam"):
        if space is None:
            space = [0, 1]
        if space != [0, 1]:
            raise Exception('For Bool no space or space of [0, 1] must be given')
        super(Bool, self).__init__(space, projection_fn, name)

class Choice(Param):
    def __init__(self, space=None, projection_fn=None, name="ChoiceParam"):
        if space is None:
            space = [0, 1]
        super(Choice, self).__init__(space, projection_fn, name)

    def sample(self):
        return self.transform_raw_sample(self.raw_sample())

    def transform_raw_sample(self, raw_sample):
        return self.projection_fn(self.space[int(raw_sample)])

    def raw_sample(self):
        rand_item = random.randint(0, len(self.space) - 1)
        return rand_item

    def create_generator(self):
        def generator():
            for i in self.space:
                yield self.projection_fn(i)
        return generator

