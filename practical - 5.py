grid = [
    ["A", "B", "C", "G"],
    ["D", "#", "E", "X"],
    ["S", "F", "H", "I"],
]
states = ["A", "B", "C", "D", "E", "S", "F", "H", "I"]
actions = ["UP", "DOWN", "LEFT", "RIGHT"]
directions = {
    "UP": (-1, 0), "DOWN": (1, 0),
    "LEFT": (0, -1), "RIGHT": (0, 1),
}
slips = {
    "UP": ["LEFT", "RIGHT"], "DOWN": ["RIGHT", "LEFT"],
    "LEFT": ["DOWN", "UP"], "RIGHT": ["UP", "DOWN"],
}
positions = {}
for row in range(len(grid)):
    for col in range(len(grid[row])):
        positions[grid[row][col]] = (row, col)


def transitions(s, a, main_prob=0.8, danger_reward=-10, step_reward=-0.1):
    row, col = positions[s]
    side_prob = (1 - main_prob) / 2
    outcomes = [(a, main_prob)]
    outcomes += [(direction, side_prob) for direction in slips[a]]
    result = []

    for direction, prob in outcomes:
        dr, dc = directions[direction]
        nr, nc = row + dr, col + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            next_state = grid[nr][nc]
            if next_state == "#":
                next_state = s
        else:
            next_state = s

        if next_state == "G":
            reward = 10
        elif next_state == "X":
            reward = danger_reward
        else:
            reward = step_reward
        result.append((prob, next_state, reward))

    return result


def q_value(s, a, V, gamma, main_prob, danger_reward, step_reward):
    q = 0.0
    for prob, next_state, reward in transitions(
        s, a, main_prob, danger_reward, step_reward
    ):
        # G және X үшін болашақ мән 0, өйткені эпизод аяқталады.
        next_value = V.get(next_state, 0.0)
        q += prob * (reward + gamma * next_value)
    return q


def value_iteration(
    gamma=0.9, main_prob=0.8, danger_reward=-10, step_reward=-0.1
):
    theta = 0.001
    V = {s: 0.0 for s in states}
    history = [V.copy()]
    iterations = 0

    while True:
        delta = 0.0
        new_V = V.copy()

        for s in states:
            q_values = []
            for a in actions:
                q = q_value(
                    s, a, V, gamma, main_prob, danger_reward, step_reward
                )
                q_values.append(q)

            new_V[s] = max(q_values)
            delta = max(delta, abs(new_V[s] - V[s]))

        # Барлық күйлер есептелген соң ғана ескі кестені ауыстырамыз.
        V = new_V
        history.append(V.copy())
        iterations += 1
        if delta < theta:
            break

    Q = {}
    policy = {}
    for s in states:
        Q[s] = {}
        for a in actions:
            Q[s][a] = q_value(
                s, a, V, gamma, main_prob, danger_reward, step_reward
            )
        policy[s] = max(actions, key=lambda a: Q[s][a])

    return V, Q, policy, history, iterations


if __name__ == "__main__":
    V, Q, policy, history, count = value_iteration()
    print("Негізгі жағдай. Итерация саны:", count)
    for s in states:
        values = ", ".join(f"{a}={Q[s][a]:.4f}" for a in actions)
        print(f"{s}: V={V[s]:.4f}, әрекет={policy[s]}; {values}")

    print("\nҚолмен есептелетін күйлер: V0, V1, V2")
    for s in ["C", "E", "H", "I"]:
        print(s, [round(history[k][s], 4) for k in range(3)])

    experiments = [
        ("1 gamma 0.2", {"gamma": 0.2}),
        ("2 негізгі бағыт 0.6", {"main_prob": 0.6}),
        ("3 қауіпті күй -30", {"danger_reward": -30}),
        ("4 қалыпты қадам -1", {"step_reward": -1}),
    ]
    for name, params in experiments:
        exp_V, _, exp_policy, _, exp_count = value_iteration(**params)
        print(f"\n{name}: итерация={exp_count}, V(S)={exp_V['S']:.4f}")
        changed = False
        for s in states:
            if policy[s] != exp_policy[s]:
                print(f"{s}: {policy[s]} -> {exp_policy[s]}")
                changed = True
        if not changed:
            print("Саясат өзгерген жоқ.")