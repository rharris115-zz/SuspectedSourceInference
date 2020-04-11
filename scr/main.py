import matplotlib.pyplot as plt
import numpy as np
from simpy import Environment

from generate import gravity_model_contact_events, infection_events, uniform_population
from model import Agent, State
from celluloid import Camera


def main():
    agents = [Agent(f'agent_{i}') for i in range(1000)]
    rng = np.random.default_rng()

    positions = uniform_population(n=len(agents), rng=rng)

    env = Environment()

    # Create patient 0.
    source = agents[len(agents) // 2]
    env.process(infection_events(env=env, infected=source, rng=rng))

    contact_events = gravity_model_contact_events(event_rate_per_agent=8.0,
                                                  exponent=2.5,
                                                  agents=agents,
                                                  positions=positions,
                                                  env=env, rng=rng)
    env.process(generator=contact_events)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(aspect='equal')
    camera = Camera(fig)

    def snap_shots():
        x, y = positions[:, 0], positions[:, 1]

        state_colours = {
            State.SUSCEPTIBLE: 'C2',
            State.INFECTED: 'C8',
            State.INFECTIOUS: 'C1',
            State.SYMPTOMATIC_INFECTIOUS: 'C3',
            State.REMOVED: 'C7'
        }
        while True:
            yield env.timeout(delay=1)
            for state, colour in state_colours.items():
                state_indices = [i for i, agent in enumerate(agents) if agent.state == state]
                ax.scatter(x=x[state_indices], y=y[state_indices], c=colour, label=state.name)
            camera.snap()

    # plt.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), prop={'size': 5})

    env.process(snap_shots())

    env.run(until=10)
    animation = camera.animate()
    # Need to have imagemagick installed. Brew is good on Mac.
    animation.save('pandemic.gif', writer='imagemagick', fps=1)


if __name__ == '__main__':
    main()
