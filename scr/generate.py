from typing import List

import numpy as np
import simpy

from model import State, Agent, get_infected


def disease_lifecycle(n: int, rng: np.random.Generator) -> np.array:
    original_positions = rng.random((n, 2))
    return original_positions


# From the Ferguson paper...
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

    if rng.uniform() < 0.5:
        # Asymptomatic
        yield env.timeout(delay=rng.normal(loc=6.5, scale=0.4))
        print(f'@{env.now} - {infected}->{State.REMOVED.name}')
        infected.state = State.REMOVED
    else:
        # Symptomatic
        yield env.timeout(delay=0.5)
        print(f'@{env.now} - {infected}->{State.SYMPTOMATIC_INFECTIOUS.name}')
        infected.state = State.SYMPTOMATIC_INFECTIOUS

        yield env.timeout(delay=rng.normal(loc=6.0, scale=0.4))
        print(f'@{env.now} - {infected}->{State.REMOVED.name}')
        infected.state = State.REMOVED


def erdos_renyi_contact_events(env: simpy.Environment, event_rate_per_agent: float, agents: List[Agent],
                               rng: np.random.Generator):
    while True:
        yield env.timeout(delay=rng.exponential(scale=event_rate_per_agent / len(agents) / 2))

        a1, a2 = rng.choice(a=agents, size=2, replace=False)
        infected = get_infected(a1=a1, a2=a2)
        if infected is not None:
            env.process(generator=infection_events(env=env, infected=infected, rng=rng))
