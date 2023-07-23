import numpy as np


class TrajectoryGenerator:

    def __init__(self, sequence_length, border_region, box_width, box_height, batch_size) -> None:
        self.sequence_length = sequence_length
        self.border_region = border_region
        self.box_width = box_width
        self.box_height = box_height
        self.batch_size = batch_size

    def avoid_wall(self, position, head_dir):
        """Compute distance and angle to nearest wall."""
        x = position[:, 0]
        y = position[:, 1]
        dists = [self.box_width / 2 - x, self.box_height / 2 - y, self.box_width / 2 + x, self.box_height / 2 + y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        head_dir = np.mod(head_dir, 2 * np.pi)
        a_wall = head_dir - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi # periodic variable

        # too close if agent is closer than [self.border_region] meters to the wall and its head direction is 
        # smaller than 90° w.r.t. the normal vector of the wall
        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(head_dir)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (np.pi / 2 - np.abs(a_wall[is_near_wall]))

        return is_near_wall, turn_angle

    def generate_trajectory(self):
        """Generate a random walk in a rectangular box."""

        samples = self.sequence_length
        dt = 0.02  # time step increment (seconds)
        b = 0.13 * 2 * np.pi # forward velocity Rayleigh distribution scale
        mu = 0  # rotation velocity Gaussian distribution mean (rad/s)
        sigma = 5.76 * 2   # rotation velocity Gaussian distribution stdev (rads/s)

        # Initialize variables
        position = np.zeros([self.batch_size, samples, 2])
        head_dir = np.zeros([self.batch_size, samples])
        position[:, 0, 0] = np.random.uniform(-self.box_width / 2, self.box_width / 2, self.batch_size)
        position[:, 0, 1] = np.random.uniform(-self.box_width / 2, self.box_width / 2, self.batch_size)
        head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, self.batch_size)
        velocity = np.zeros([self.batch_size, samples])

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [self.batch_size, samples + 1])
        random_vel = np.random.rayleigh(b, [self.batch_size, samples + 1])

        for t in range(samples - 1):
            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(self.batch_size)

            is_near_wall, turn_angle = self.avoid_wall(position[:, t], head_dir[:, t])
            v[is_near_wall] *= 0.25 # slow down
                
            # Update turn angle
            turn_angle += dt * random_turn[:, t]

            # Take a step
            velocity[:, t] = v * dt
            update = velocity[:, t, None] * np.stack([np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1)
            position[:, t + 1] = position[:, t] + update

            # Rotate head direction
            head_dir[:, t + 1] = head_dir[:, t] + turn_angle

        head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi # periodic variable
    
        return position, velocity, head_dir
