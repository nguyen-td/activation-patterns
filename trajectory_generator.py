import numpy as np


class TrajectoryGenerator:

    def __init__(self, sequence_length, border_region) -> None:
        self.sequence_length = sequence_length
        self.border_region = border_region

    def avoid_wall(self, position, head_dir, box_width, box_height):
        """Compute distance and angle to nearest wall."""
        x = position[0]
        y = position[1]
        dists = [box_width / 2 - x, box_height / 2 - y, box_width / 2 + x, box_height / 2 + y]
        d_wall = np.min(dists)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists)]
        head_dir = np.mod(head_dir, 2 * np.pi)
        a_wall = head_dir - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi # periodic variable

        # too close if agent is closer than [self.border_region] meters to the wall and its head direction is 
        # smaller than 90° w.r.t. the normal vector of the wall
        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = 0
        if is_near_wall:
            turn_angle = np.sign(a_wall) * (np.pi / 2 - np.abs(a_wall))

        return is_near_wall, turn_angle

    def generate_trajectory(self, box_width, box_height):
        """Generate a random walk in a rectangular box."""

        samples = self.sequence_length
        dt = 0.02  # time step increment (seconds)
        b = 0.13 * 2 * np.pi # forward velocity Rayleigh distribution scale
        mu = 0  # rotation velocity Gaussian distribution mean (rad/s)
        sigma = 5.76 * 2   # rotation velocity Gaussian distribution stdev (rads/s)

        # Initialize variables
        position = np.zeros([samples, 2])
        head_dir = np.zeros([samples])
        position[0, 0] = np.random.uniform(-box_width / 2, box_width / 2)
        position[0, 1] = np.random.uniform(-box_width / 2, box_width / 2)
        head_dir[0] = np.random.uniform(0, 2 * np.pi)
        velocity = np.zeros([samples])

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, samples + 1)
        random_vel = np.random.rayleigh(b, samples + 1)

        for t in range(samples - 1):
            # Update velocity
            v = random_vel[t]
            turn_angle = 0

            is_near_wall, turn_angle = self.avoid_wall(position[t], head_dir[t], box_width, box_height)
            if is_near_wall:
                v *= 0.25 # slow down
                
            # Update turn angle
            turn_angle += dt * random_turn[t]

            # Take a step
            velocity[t] = v * dt
            update = velocity[t] * np.stack([np.cos(head_dir[t]), np.sin(head_dir[t])], axis=-1)
            position[t + 1] = position[t] + update

            # Rotate head direction
            head_dir[t + 1] = head_dir[t] + turn_angle

        head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi # periodic variable
    
        return position, velocity, head_dir
