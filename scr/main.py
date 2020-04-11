import matplotlib.pyplot as plt
import numpy as np
from simpy import Environment

from generate import gravity_model_contact_events, infection_events, uniform_population
from model import Agent, State
from celluloid import Camera
from typing import Dict, List


def main():
    agents = [Agent(f'agent_{i}') for i in range(1000)]
    rng = np.random.default_rng()

    positions = uniform_population(n=len(agents), rng=rng)

    env = Environment()

    # Create patient 0.
    source = agents[len(agents) // 2]
    env.process(infection_events(env=env, infected=source, rng=rng))

    contact_events = gravity_model_contact_events(event_rate_per_agent=4.0,
                                                  exponent=1.75,
                                                  agents=agents,
                                                  positions=positions,
                                                  env=env, rng=rng)
    env.process(generator=contact_events)

    fig = plt.figure()
    pop_ax = fig.add_subplot(1, 2, 1)
    tot_ax = fig.add_subplot(1, 2, 2)
    pop_ax.set_aspect(aspect='equal')
    camera = Camera(fig)

    until = 200

    def snap_shots():
        x, y = positions[:, 0], positions[:, 1]

        state_colours = {
            State.SUSCEPTIBLE: '#2ca02c',
            State.INFECTED: '#bcbd22',
            State.INFECTIOUS: '#ff7f0e',
            State.SYMPTOMATIC_INFECTIOUS: '#d62728',
            State.REMOVED: '#7f7f7f'
        }

        totals: Dict[State, List[int]] = {state: [] for state in State}
        times: List[float] = []
        while True:
            yield env.timeout(delay=1)
            for state, colour in state_colours.items():
                state_indices = [i for i, agent in enumerate(agents) if agent.state == state]
                pop_ax.scatter(x=x[state_indices], y=y[state_indices], c=colour, label=state.name, s=2)
                totals[state].append(len(state_indices))

            times.append(env.now)
            tot_ax.stackplot(times, [ts for ts in totals.values()],
                             colors=[state_colours[state] for state in totals.keys()])
            plt.legend([state.name for state in totals.keys()], loc='lower left', prop={'size': 5})
            camera.snap()

            infected_count = sum(total[-1] for state, total in totals.items() if state.active())
            if env.now > 0 and infected_count == 0:
                print('No longer taking snaps.')
                break

    env.process(snap_shots())

    env.run(until=until)
    animation = camera.animate()
    # Need to have imagemagick installed. Brew is good on Mac.
    animation.save('pandemic.gif', writer='imagemagick', fps=1)


if __name__ == '__main__':
    main()
