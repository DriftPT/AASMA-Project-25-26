from mesa import Agent

from src.utils import manhattan_distance, move_towards


class ZombieAgent(Agent):
    """
    Zombie agent.

    Behaviour:
    - attacks if already adjacent to a survivor;
    - chases survivors only if they are within zombie_vision_range;
    - otherwise performs a smarter random walk.
    """

    def __init__(self, model):
        super().__init__(model)

        self.health = 2
        self.max_health = 2
        self.alive = True

        # Used for smarter wandering.
        self.visited_positions = set()
        self.previous_pos = None
        self.wander_direction = None
        self.wander_steps_left = 0

    def step(self):
        if not self.alive or self.model.finished:
            return

        survivors = self.model.get_active_survivors()

        if not survivors:
            return

        self.visited_positions.add(self.pos)

        # 1. Attack if already adjacent.
        adjacent_survivors = [
            survivor for survivor in survivors
            if manhattan_distance(self.pos, survivor.pos) <= self.model.attack_range
        ]

        if adjacent_survivors:
            target = min(
                adjacent_survivors,
                key=lambda survivor: survivor.health
            )

            target.take_damage(1)
            return

        # 2. Chase only visible survivors.
        visible_survivors = [
            survivor for survivor in survivors
            if manhattan_distance(self.pos, survivor.pos) <= self.model.zombie_vision_range
        ]

        if visible_survivors:
            target = min(
                visible_survivors,
                key=lambda survivor: manhattan_distance(self.pos, survivor.pos)
            )

            new_pos = move_towards(
                self.pos,
                target.pos,
                rng=self.model.random
            )

            if self.model.can_move_to(new_pos, moving_agent=self):
                self.previous_pos = self.pos
                self.model.grid.move_agent(self, new_pos)

            # Reset wandering direction when chasing.
            self.wander_direction = None
            self.wander_steps_left = 0
            return

        # 3. If no survivor is visible, wander intelligently.
        self.smart_random_walk()

    def smart_random_walk(self):
        """
        Smarter wandering behaviour.

        The zombie:
        - avoids immediately going back to the previous cell if possible;
        - prefers less visited cells;
        - sometimes keeps walking in the same direction;
        - still has randomness, so movement does not look deterministic.
        """
        candidates = self.get_valid_neighbour_positions()

        if not candidates:
            return

        # 1. Try to continue in the same direction for a few steps.
        if self.wander_direction is not None and self.wander_steps_left > 0:
            forward_pos = (
                self.pos[0] + self.wander_direction[0],
                self.pos[1] + self.wander_direction[1],
            )

            if forward_pos in candidates:
                self.move_to_wander_position(forward_pos)
                self.wander_steps_left -= 1
                return

        # 2. Avoid going back to the previous position, if possible.
        non_backtracking = [
            pos for pos in candidates
            if pos != self.previous_pos
        ]

        if non_backtracking:
            candidates = non_backtracking

        # 3. Prefer positions this zombie has not visited yet.
        unvisited = [
            pos for pos in candidates
            if pos not in self.visited_positions
        ]

        if unvisited:
            candidates = unvisited

        # 4. Choose among the best candidates randomly.
        chosen_pos = self.model.random.choice(candidates)

        # 5. Save direction for short-term movement persistence.
        self.set_wander_direction(chosen_pos)

        self.move_to_wander_position(chosen_pos)

    def get_valid_neighbour_positions(self):
        x, y = self.pos

        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        return [
            pos for pos in candidates
            if self.model.can_move_to(pos, moving_agent=self)
        ]

    def set_wander_direction(self, new_pos):
        dx = new_pos[0] - self.pos[0]
        dy = new_pos[1] - self.pos[1]

        self.wander_direction = (dx, dy)

        # Zombie keeps direction for 2 to 4 steps.
        self.wander_steps_left = self.model.random.randint(2, 4)

    def move_to_wander_position(self, new_pos):
        self.previous_pos = self.pos
        self.model.grid.move_agent(self, new_pos)