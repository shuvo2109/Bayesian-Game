import json
import random


def get_neighbors(pos):
    x, y = pos
    return {(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x, y), (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)}


class Environment:
    def __init__(self, size, mode="debug", red_dens=3/36, green_dens=2/36, obs_dens=8/36):
        self.width = size[0]
        self.height = size[1]

        self.states = [(x, y) for x in range(self.width) for y in range(self.height)]

        self.mode = mode

        self.red_dens = red_dens
        self.green_dens = green_dens
        self.obs_dens = obs_dens

        self.obstacles = []
        self.red_cells = []
        self.grn_cells = []

        self.p1_position = (0, 0)
        self.p2_position = (0, 0)

    def assign_obstacles(self):
        if self.mode.lower() == "debug":
            self.obstacles = [(3, 0), (3, 1), (0, 2), (1, 2), (4, 3), (5, 3), (2, 4), (2, 5)]
            self.red_cells = [(0, 0), (4, 1), (1, 5)]
            self.grn_cells = [(4, 0), (1, 3)]
        else:
            available_positions = list(self.states)

            # Obstacles
            num_obs = int(self.width * self.height * self.obs_dens)
            self.obstacles = random.sample(available_positions, num_obs)

            # Red Cells
            available_positions = [pos for pos in available_positions if pos not in self.obstacles]
            num_red = int(self.width * self.height * self.red_dens)
            self.red_cells = random.sample(available_positions, num_red)

            # Green Cells
            available_positions = [pos for pos in available_positions if pos not in self.red_cells]
            num_green = int(self.width * self.height * self.green_dens)
            self.grn_cells = random.sample(available_positions, num_green)

    def assign_players(self):
        if self.mode == "debug":
            self.p1_position = (2, 2)
            self.p2_position = (3, 3)
        else:
            occupied_positions = set(self.obstacles + self.red_cells + self.grn_cells)
            available_positions = [pos for pos in self.states if pos not in occupied_positions]

            # Player 1 position
            self.p1_position = random.choice(available_positions)

            # Player 2 position
            available_positions.remove(self.p1_position)
            self.p2_position = random.choice(available_positions)

    def is_valid(self, pos):
        return (0 <= pos[0] < self.width and
                0 <= pos[1] < self.height and
                pos not in self.obstacles)

    def move(self, pos, action):
        moves = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0), '0': (0, 0), 'Z': (0, 0)}
        new_pos = (pos[0] + moves[action][0], pos[1] + moves[action][1])
        return new_pos if self.is_valid(new_pos) else pos

    def save_to_json(self, filename="environment.json"):
        env_data = {
            "obstacles": self.obstacles,
            "red_cells": self.red_cells,
            "green_cells": self.grn_cells
        }
        with open(filename, "w") as file:
            json.dump(env_data, file, indent=4)

        print(f"Environment data saved to {filename}.")


class Player1:
    def __init__(self, env: Environment, alpha=0.1, gamma=0.9):
        self.actions = ['0', 'N', 'E', 'W', 'S', 'Z']
        self.position = env.p1_position

        self.belief = {0: 0.5, 1: 0.5}

        p2_actions = ['0', 'N', 'E', 'W', 'S']

        self.p_a2_given_theta2 = {x: {a: 1/len(p2_actions) for a in p2_actions} for x in [0, 1]}

        self.VI = {a: 0 for a in self.actions}
        self.Q = {(a1, a2): 0 for a1 in self.actions for a2 in p2_actions}

        self.alpha = alpha
        self.gamma = gamma

    def p_history_given_theta2(self, history, theta2):
        p = 1

        for a2 in history['a2']:
            p *= self.p_a2_given_theta2[theta2][a2]

        return p

    def update_belief(self, history):
        likelihood = {x: self.p_history_given_theta2(history, x) for x in [0, 1]}
        prior = self.belief
        evidence = sum(likelihood[x] * prior [x] for x in [0, 1])

        for x in [0, 1]:
            self.belief[x] = (likelihood[x] * prior[x]) / evidence if evidence > 0 else prior[x]

    def choose_action(self, p2_actions):
        for a1 in self.actions:
            self.VI[a1] = sum(self.belief[x] * sum(self.Q[(a1, a2)] * self.p_a2_given_theta2[x][a2] for a2 in p2_actions) for x in [0, 1])

        return max(self.VI, key=self.VI.get)

    def Q_update(self, history, a1t, a2t, r1):
        max_next_Q = max(self.VI[a1] for a1 in self.actions)

        expected_value = sum(self.p_history_given_theta2(history, x) * max_next_Q for x in [0, 1])

        self.Q[(a1t, a2t)] += self.alpha * (r1 + self.gamma * expected_value - self.Q[(a1t, a2t)])


class Player2:
    def __init__(self, env: Environment, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.actions = ['0', 'N', 'E', 'W', 'S']
        self.position = env.p2_position

        self.theta2 = 0
        self.epsilon = epsilon

        self.Q = {(p1, p2): {a: 0 for a in self.actions} for p1 in env.states for p2 in env.states}

        self.alpha = alpha
        self.gamma = gamma

    def assign_type(self, prob_cooperative=0.5):
        self.theta2 = 0 if random.random() < prob_cooperative else 1

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)

        q_values = self.Q[state]

        return max(q_values, key=q_values.get)

    def Q_update(self, st, a2t, r2, next_st):
        max_next_Q = max(self.Q[next_st][a2] for a2 in self.actions)  # Max future Q-value

        # Q-learning update
        self.Q[st][a2t] += self.alpha * (r2 + self.gamma * max_next_Q - self.Q[st][a2t])


class Game:
    def __init__(self, env: Environment, player1: Player1, player2: Player2, B: float, C: float):
        self.env = env
        self.p1 = player1
        self.p2 = player2
        self.history = {'s': [], 'a1': [], 'a2': []}

        self.B = B  # Competition reward / punishment
        self.C = C  # False Negative Punishment

        self.trajectory = {}

    def run(self, num_episodes=1000, T=10):
        for episode in range(num_episodes):
            # Episode initialization
            self.p1.update_belief(self.history)
            self.p2.assign_type()
            self.history = {'s': [], 'a1': [], 'a2': []}

            self.env.assign_players()

            self.p1.position = self.env.p1_position
            self.p2.position = self.env.p2_position

            pos1 = self.env.p1_position
            pos2 = self.env.p2_position

            self.history['s'].append((pos1, pos2))

            # Record Trajectory
            episode_data = {
                "belief": self.p1.belief.copy(),
                "turns": []
            }

            for t in range(T):
                # Turn initialization
                st = self.history['s'][-1]

                p1t, p2t = st[0], st[1]

                r1, r2 = 0, 0

                # Action Choice
                a1t = self.p1.choose_action(self.p2.actions)
                a2t = self.p2.choose_action(state=st)

                # Transition
                p1t_new = self.env.move(pos=p1t, action=a1t)
                p2t_new = self.env.move(pos=p2t, action=a2t)

                # If P1 uses zap while P2 is in the vicinity, the game ends
                # If P2 is cooperative agent, zapping is bad for both players
                if a1t == "Z":
                    if p2t_new in get_neighbors(pos=p1t_new):
                        if self.p2.theta2 == 0:
                            r1, r2 = -self.C, -self.C

                        self.p1.Q_update(self.history, a1t, a2t, r1)
                        self.p2.Q_update(st, a2t, r2, (p1t_new, p2t_new))

                        # Record this turn
                        episode_data["turns"].append({
                            "p1t": p1t, "p2t": p2t,
                            "a1t": a1t, "a2t": a2t,
                            "r1": r1, "r2": r2,
                            "Q1": self.p1.Q[(a1t, a2t)],
                            "Q2": self.p2.Q[st][a2t],
                            "VI1": self.p1.VI.copy()
                        })


                        self.trajectory[episode] = episode_data

                        break

                # Cooperative agent reaching green cell sharing mutual rewards
                if self.p2.theta2 == 0 and p2t_new in self.env.grn_cells:
                    r1, r2 = 1, 1

                # Competitive agent reaching red cell stealing rewards
                if self.p2.theta2 == 1 and p2t_new in self.env.red_cells:
                    r1, r2 = -self.B, self.B

                # Q-update
                self.p1.Q_update(self.history, a1t, a2t, r1)
                self.p2.Q_update(st, a2t, r2, (p1t_new, p2t_new))

                # History update
                self.history['s'].append((p1t_new, p2t_new))
                self.history['a1'].append(a1t)
                self.history['a2'].append(a2t)

                # Record this turn
                episode_data["turns"].append({
                    "p1t": p1t, "p2t": p2t,
                    "a1t": a1t, "a2t": a2t,
                    "r1": r1, "r2": r2,
                    "Q1": self.p1.Q[(a1t, a2t)],
                    "Q2": self.p2.Q[st][a2t],
                    "VI1": self.p1.VI.copy()
                })

            self.trajectory[episode] = episode_data

    def export_trajectory(self, filename="trajectory.json"):
        with open(filename, "w") as f:
            json.dump(self.trajectory, f, indent=2)


