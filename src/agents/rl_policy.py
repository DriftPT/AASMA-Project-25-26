class RLPolicy:
    """
    Stores and updates the Q-table used by AdaptiveAgent.
    Supports Q-Learning (Off-Policy) and SARSA (On-Policy) algorithms.
    """

    def __init__(
        self,
        actions,
        algorithm="q_learning", # "q_learning" or "sarsa"
        alpha=0.2,
        gamma=0.9,
        epsilon=1,
        epsilon_decay=0.999,    # Multiplier for decay at the end of the run
        training=True,
    ):
        self.actions = tuple(actions)
        self.algorithm = algorithm.lower()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.training = training

        self.q_table = {}

    def set_training(self, training: bool):
        self.training = training

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def decay_epsilon(self):
        """Applies epsilon decay (typically called at the end of an episode)."""
        if self.training:
            self.epsilon *= self.epsilon_decay

    def choose_action(self, state, rng):
        """Chooses action using epsilon-greedy strategy."""
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

    def update(self, state, action, reward, next_state, next_action=None):
        """
        Applies the selected update rule (Q-learning or SARSA).
        """
        if not self.training:
            return

        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)

        current_q = self.q_table[state][action]

        if self.algorithm == "q_learning":
            # Uses the optimal estimate (best possible action in the next state)
            target_q = max(self.q_table[next_state].values())
            
        elif self.algorithm == "sarsa":
            # Uses the action that the epsilon-greedy policy actually chose
            if next_action is None:
                raise ValueError("SARSA algorithm requires the 'next_action' parameter.")
            target_q = self.q_table[next_state][next_action]
            
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Generic update formula (TD target)
        new_q = current_q + self.alpha * (reward + self.gamma * target_q - current_q)

        self.q_table[state][action] = new_q

    def ensure_state_exists(self, state):
        if state not in self.q_table:
            self.q_table[state] = {
                action: 0.0
                for action in self.actions
            }

    def number_of_learned_states(self):
        return len(self.q_table)