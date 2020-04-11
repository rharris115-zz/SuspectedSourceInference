from typing import List

import numpy as np
import simpy
from scipy.spatial import KDTree

from model import State, Agent, get_infected


def uniform_population(n: int, rng: np.random.Generator) -> np.array:
    return rng.random((n, 2))


# From the Ferguson paper...
# We assumed an incubation period of 5.1 days. Infectiousness is assumed to occur from 12 hours
# prior to the onset of symptoms for those that are symptomatic and from 4.6 days after infection in
# those that are asymptomatic with an infectiousness profile over time that results in a 6.5-day mean
# generation time.

def infection_events(env: simpy.Environment, infected: Agent, rng: np.random.Generator):
    print(f'@t={env.now} - {infected}->{State.INFECTED.name}')
    infected.state = State.INFECTED
    yield env.timeout(delay=rng.normal(loc=4.6, scale=0.3))

    print(f'@t={env.now} - {infected}->{State.INFECTIOUS.name}')
    infected.state = State.INFECTIOUS

    if rng.uniform() < 0.5:
        # Asymptomatic
        yield env.timeout(delay=rng.normal(loc=6.5, scale=0.4))
        print(f'@t={env.now} - {infected}->{State.REMOVED.name}')
        infected.state = State.REMOVED
    else:
        # Symptomatic
        yield env.timeout(delay=0.5)
        print(f'@t={env.now} - {infected}->{State.SYMPTOMATIC_INFECTIOUS.name}')
        infected.state = State.SYMPTOMATIC_INFECTIOUS

        yield env.timeout(delay=rng.normal(loc=6.0, scale=0.4))
        print(f'@t={env.now} - {infected}->{State.REMOVED.name}')
        infected.state = State.REMOVED


def gravity_model_contact_events(event_rate_per_agent: float,
                                 exponent: float,
                                 agents: List[Agent],
                                 positions: np.array,
                                 env: simpy.Environment,
                                 rng: np.random.Generator):
    tree = KDTree(data=positions)
    close_pairs = list(tree.query_pairs(r=0.1))
    inverse_distances = np.array([np.linalg.norm(positions[idx1] - positions[idx2]) ** -exponent
                                  for idx1, idx2 in close_pairs])
    inverse_distances /= inverse_distances.sum()

    while True:
        choices = rng.choice(a=close_pairs, p=inverse_distances, size=len(agents)).tolist()
        for choice in choices:
            yield env.timeout(delay=rng.exponential(scale=1 / len(agents) / event_rate_per_agent))
            contact_agents = [agents[idx] for idx in choice]
            infected = get_infected(contact_agents)
            for i in infected:
                env.process(generator=infection_events(env=env, infected=i, rng=rng))


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
