class QLearningPolicy:
    """
    Stores and updates the Q-table used by the AdaptiveAgent.

    The Q-table is hidden inside this object.
    The experiments do not need to access q_table directly.
    """

    def __init__(
        self,
        actions,
        alpha=0.2,
        gamma=0.9,
        epsilon=0.25,
        training=True,
    ):
        self.actions = tuple(actions)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

        self.q_table = {}

    def set_training(self, training: bool):
        self.training = training

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def choose_action(self, state, rng):
        """
        Chooses an action using epsilon-greedy.

        During training:
        - sometimes chooses a random action to explore.

        During testing:
        - epsilon should be 0, so it chooses the best known action.
        """
        self.ensure_state_exists(state)

        if self.training and rng.random() < self.epsilon:
            return rng.choice(list(self.actions))

        q_values = self.q_table[state]
        best_value = max(q_values.values())

        best_actions = [
            action
            for action, value in q_values.items()
            if value == best_value
        ]

        return rng.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """
        Applies the Q-learning update rule.

        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
        """
        if not self.training:
            return

        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)

        current_q = self.q_table[state][action]
        best_next_q = max(self.q_table[next_state].values())

        new_q = current_q + self.alpha * (
            reward + self.gamma * best_next_q - current_q
        )

        self.q_table[state][action] = new_q

    def ensure_state_exists(self, state):
        if state not in self.q_table:
            self.q_table[state] = {
                action: 0.0
                for action in self.actions
            }

    def number_of_learned_states(self):
        return len(self.q_table)