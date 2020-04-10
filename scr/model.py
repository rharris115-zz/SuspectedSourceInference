from enum import IntEnum, auto, unique
from typing import Union


@unique
class State(IntEnum):
    SUSCEPTIBLE = auto()
    INFECTED = auto()
    INFECTIOUS = auto()
    SYMPTOMATIC_INFECTIOUS = auto()
    REMOVED = auto()

    def infectious(self):
        return self == State.INFECTIOUS or self == State.SYMPTOMATIC_INFECTIOUS

    def susceptible(self):
        return self == State.SUSCEPTIBLE


class Agent:

    def __init__(self, name: str):
        self._name: str = name
        self._state: State = State.SUSCEPTIBLE

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, value: State):
        assert self._state < value, f'Illegal state transition. {self._state} -> {value}'
        self._state = value

    @classmethod
    def infected(cls, a1, a2):
        pass

    def __repr__(self):
        return f'{self.name}: {self.state.name}'


def get_infected(a1: Agent, a2: Agent) -> Union[Agent, type(None)]:
    s1, s2 = a1.state, a2.state
    if s1.infectious() and s2.susceptible():
        return a2
    elif s1.susceptible() and s2.infectious():
        return a1
    else:
        return None
