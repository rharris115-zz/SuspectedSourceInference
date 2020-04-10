from generate import erdos_renyi_contact_events, infection_events
from model import Agent, State
from simpy import Environment
import numpy as np


def main():
    agents = [Agent(f'agent_{i}') for i in range(1000)]
    rng = np.random.default_rng()

    env = Environment()

    source = agents[len(agents) // 2]
    env.process(infection_events(env=env, infected=source, rng=rng))
    contact_events = erdos_renyi_contact_events(env=env, event_rate_per_agent=12.0, agents=agents, rng=rng)
    env.process(generator=contact_events)
    env.run(until=1000)


if __name__ == '__main__':
    main()
