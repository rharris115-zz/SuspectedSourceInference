from typing import List

import numpy as np
import simpy
from scipy.spatial import KDTree

from model import State, Agent, get_infected
from contact_parameters import n_individuals, contact_rate_gravity_exponent, \
    contact_rate_per_individual, p_symptomatic_individual_isolates, contact_distance_upper_bound
from infection_parameters import p_asymptomatic


def uniform_population(rng: np.random.Generator) -> np.array:
    # Distribute population uniformly on a unit square.
    return rng.random((n_individuals, 2))


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

    if rng.uniform() < p_asymptomatic:
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


def gravity_model_contact_events(agents: List[Agent],
                                 positions: np.array,
                                 env: simpy.Environment,
                                 rng: np.random.Generator):
    tree = KDTree(data=positions)
    close_pairs = list(tree.query_pairs(r=contact_distance_upper_bound))
    inverse_distances = np.array(
        [np.linalg.norm(positions[idx1] - positions[idx2]) ** -contact_rate_gravity_exponent
         for idx1, idx2 in close_pairs])
    inverse_distances /= inverse_distances.sum()

    while True:
        choices = rng.choice(a=close_pairs, p=inverse_distances, size=len(agents)).tolist()
        for choice in choices:
            yield env.timeout(delay=rng.exponential(scale=1 / len(agents) / contact_rate_per_individual))
            contact_agents = [agents[idx]
                              for idx in choice
                              if not agents[idx].state.symptomatic()
                              or rng.uniform() > p_symptomatic_individual_isolates]

            if len(contact_agents) < 2:
                # Symptomatic self-isolation means this is no longer a contact event and doesn't need
                # recording. Skip to the next event.
                continue

            infected = get_infected(contact_agents, rng=rng)
            for i in infected:
                env.process(generator=infection_events(env=env, infected=i, rng=rng))
