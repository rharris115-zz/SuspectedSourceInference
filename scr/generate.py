from typing import List, Iterable, Any, Tuple
import numpy as np
from model import State, Agent
import simpy


# We assumed an incubation period of 5.1 days. Infectiousness is assumed to occur from 12 hours
# prior to the onset of symptoms for those that are symptomatic and from 4.6 days after infection in
# those that are asymptomatic with an infectiousness profile over time that results in a 6.5-day mean
# generation time.

def infection_events(env: simpy.Environment, infected: Agent, rng: np.random.Generator):
    print(f'@{env.now} - {infected}->{State.INFECTED.name}')
    infected.state = State.INFECTED

    yield env.timeout(delay=rng.normal(loc=4.6, scale=0.3))
    print(f'@{env.now} - {infected}->{State.INFECTIOUS.name}')
    infected.state = State.INFECTIOUS

    yield env.timeout(delay=rng.normal(loc=6.5, scale=0.4))
    print(f'@{env.now} - {infected}->{State.REMOVED.name}')
    infected.state = State.REMOVED


def erdos_renyi_contact_events(env: simpy.Environment, event_rate_per_agent: float, agents: List[Agent],
                               rng: np.random.Generator):
    while True:
        a1, a2 = rng.choice(a=agents, size=2, replace=False)
        yield env.timeout(delay=rng.exponential(scale=event_rate_per_agent / len(agents) / 2))

        s1, s2 = a1.state, a2.state
        print(f'@{env.now} - "{a1}" -- "{a2}"')
        if s1 == State.INFECTIOUS and s2 == State.SUSCEPTIBLE:
            env.process(generator=infection_events(env=env, infected=a2, rng=rng))
        elif s1 == State.SUSCEPTIBLE and s2 == State.INFECTIOUS:
            env.process(generator=infection_events(env=env, infected=a1, rng=rng))
        else:
            pass
