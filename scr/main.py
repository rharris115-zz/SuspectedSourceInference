import numpy as np
from simpy import Environment
from generate import gravity_model_contact_events, infection_events, uniform_population
from model import Agent


def main():
    agents = [Agent(f'agent_{i}') for i in range(1000)]
    rng = np.random.default_rng()

    positions = uniform_population(n=len(agents), rng=rng)

    env = Environment()

    # Create patient 0.
    source = agents[len(agents) // 2]
    env.process(infection_events(env=env, infected=source, rng=rng))

    contact_events = gravity_model_contact_events(event_rate_per_agent=2.0,
                                                  exponent=2,
                                                  agents=agents,
                                                  positions=positions,
                                                  env=env, rng=rng)
    env.process(generator=contact_events)
    env.run(until=1000)


if __name__ == '__main__':
    main()
