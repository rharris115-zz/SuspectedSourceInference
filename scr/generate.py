from typing import List

import numpy as np
import simpy

from model import State, Agent, get_infected


def uniform_population(n: int, rng: np.random.Generator) -> np.array:
    return rng.random((n, 2))


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


def erdos_renyi_contact_events(env: simpy.Environment,
                               event_rate_per_agent: float,
                               agents: List[Agent],
                               rng: np.random.Generator):
    while True:

        contact_agents = rng.choice(a=agents, size=2, replace=False).tolist()

        yield env.timeout(delay=rng.exponential(scale=event_rate_per_agent / len(agents) / 2))

        infected = get_infected(contact_agents)
        for i in infected:
            env.process(generator=infection_events(env=env, infected=i, rng=rng))
