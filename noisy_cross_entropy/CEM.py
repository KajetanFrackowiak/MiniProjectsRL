import numpy as np
import gymnasium as gym
import yaml

def load_hyperparameters(method, env_name):
    with open('hyperparameters.yaml', 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams.get(method, {}).get(env_name, {})

class NoisyCrossEntropy:
    def __init__(self, num_features, population_size, elite_ratio, noise_std, initial_std, decreasing_noise=False):
        self.num_features = num_features
        self.population_size = population_size
        self.num_elite = max(1, int(population_size * elite_ratio))
        self.initial_noise_std = noise_std
        self.noise_std = noise_std
        self.decreasing_noise = decreasing_noise

        self.mean = np.zeros(num_features)
        self.std = np.full(num_features, initial_std)

        self.generation = 0
        self.best_weights = None
        self.best_score = -np.inf
        self.score_history = []
    
    def sample_population(self):
        samples = np.random.normal(
            loc=self.mean,
            scale=self.std,
            size=(self.population_size, self.num_features)
        )
        return samples
    
    def add_noise_to_elite(self, elite_weights):
        noise = np.random.normal(0, self.noise_std, elite_weights.shape)
        noisy_elite = elite_weights + noise
        return noisy_elite
    
    def update_distribution(self, population, scores):
        # Update noise schedule first
        self.update_noise_schedule()
        
        threshold = np.percentile(scores, 100 * (1 - self.num_elite / self.population_size))
        selected_samples = population[scores >= threshold]

        noise = np.random.normal(0, self.noise_std, selected_samples.shape)
        noisy_samples = selected_samples + noise

        # noisy_samples.shape = (population_size, num_features)
        new_mean = np.mean(noisy_samples, axis=0)
        new_std = np.std(noisy_samples, axis=0)

        self.mean = new_mean
        self.std = new_std

        best_idx = np.argmax(scores)
        if scores[best_idx] > self.best_score:
            self.best_score = scores[best_idx]
            self.best_weights = population[best_idx].copy()
        
        self.score_history.append(self.best_score)
        self.generation += 1

        self.update_noise_schedule()  # Update noise schedule after each generation
    
    def get_best_weights(self):
        return self.best_weights.copy() if self.best_weights is not None else None

    def get_stats(self):
        return {
            'generation': self.generation,
            'best_score': self.best_score,
            'current_noise_std': self.noise_std,
            'mean': self.mean.copy(),
            'std': self.std.copy(),
            'score_history': self.score_history
        }

    def update_noise_schedule(self):
        """Update noise standard deviation based on generation using decreasing schedule.
        Formula from Szita & Lőrincz: Zt = max(5 - t/10, 0)
        """
        if self.decreasing_noise:
            # Using the formula from the paper: max(5 - t/10, 0) scaled by initial noise
            noise_factor = max(5 - self.generation / 10, 0) / 5.0  # Normalize to [0, 1]
            self.noise_std = self.initial_noise_std * noise_factor
        else:
            self.noise_std = self.initial_noise_std
        

class CEMPolicy:
    def __init__(self, obs_dim, act_dim, discrete=True):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.discrete = discrete
        self.weights = np.random.randn(obs_dim, act_dim) * 0.1

    def set_weights(self, weights):
        self.weights = weights.reshape(self.obs_dim, self.act_dim)
    
    def get_weights_flat(self):
        return self.weights.flatten()

    def act(self, obs):
        if self.discrete:
            # logits.shape = (self.obs_dim, self.act_dim)
            logits = np.dot(obs, self.weights)
            if self.act_dim == 1:
                return int(logits[0] > 0)  # For binary actions, return 0 or 1 as int
            else:
                return np.argmax(logits)  # For multi-class actions
        else:
            action = np.dot(obs, self.weights)
            return np.clip(action, -2.0, 2.0)  # For continuous actions
        

class CEMTrainer:
    def __init__(self, env_name, population_size, elite_ratio, seed, method, decreasing_noise=False):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.method = method

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.discrete = True
            # For discrete action spaces, we assume the action space is either binary or multi-class
            self.act_dim = self.env.action_space.n
        else:
            self.discrete = False
            # For continuous action spaces, we assume the action space is a vector
            self.act_dim = self.env.action_space.shape[0]
        
        # Observation space is assumed to be a vector
        self.obs_dim = self.env.observation_space.shape[0]
        self.num_params = self.obs_dim * self.act_dim

        self.cem = NoisyCrossEntropy(
            num_features=self.num_params,
            population_size=population_size,
            elite_ratio=elite_ratio,
            noise_std=0.1,
            initial_std=1.0,
            decreasing_noise=decreasing_noise
        )

    def evaluate_policy(self, weights, num_episodes):
        policy = CEMPolicy(self.obs_dim, self.act_dim, self.discrete)
        policy.set_weights(weights)

        total_reward = 0

        for episode in range(num_episodes):
            observation = self.env.reset()[0]
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 1000
            
            while not done and steps < max_steps:
                action = policy.act(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            # print(f"Episode reward: {episode_reward} after {steps} steps.")

            total_reward += episode_reward
        # print(f"Total reward: {total_reward} over {num_episodes} episodes.")

        return total_reward / num_episodes
    

    def train(self, num_generations, num_episodes):
        hyperparameters = load_hyperparameters(method=self.method, env_name=self.env_name)
        scores_history = []

        for generation in range(num_generations):
            population = self.cem.sample_population()

            scores = []
            for weights in population:
                score = self.evaluate_policy(weights, num_episodes)
                scores.append(score)
            
            scores = np.array(scores)
            scores_history.append(scores)

            self.cem.update_distribution(population, scores)

            # if generation % 10 == 0:
            stats = self.cem.get_stats()
            print(f"Number of scores: {len(scores)}")
            print(f"Generation {generation} | Best Score: {self.cem.best_score} | Mean: {np.mean(scores):.4f} | Overall Best: {stats['best_score']:.4f} | Noise Std: {stats['current_noise_std']:.4f} | Seed: {self.seed}")
            if self.env_name == "CartPole-v1" and np.mean(scores) >= hyperparameters.get('solve_threshold', 500):
                print(f"Solved! Reached score of {hyperparameters.get('solve_threshold'), 500} in generation {generation}.") 
                break
            elif self.env_name == "Pendulum-v1" and np.mean(scores) >= hyperparameters.get('solve_threshold', -1000):
                print(f"Solved! Reached score of {hyperparameters.get('solve_threshold', -1000)} in generation {generation}.")
                break
            elif self.env_name == "MountainCar-v0" and np.mean(scores) >= hyperparameters.get('solve_threshold', -110):
                print(f"Solved! Reached score of {hyperparameters.get('solve_threshold', -110)} in generation {generation}.")
                break
            elif self.env_name == "Acrobot-v1" and np.mean(scores) >= hyperparameters.get('solve_threshold', -100):
                print(f"Solved! Reached score of {hyperparameters.get('solve_threshold'), -100} in generation {generation}.")
                break

        best_weights = self.cem.get_best_weights()
        best_policy = CEMPolicy(self.obs_dim, self.act_dim, self.discrete)
        if best_weights is not None:
            best_policy.set_weights(best_weights)
        print("Training complete. Best policy obtained.")
        return np.mean(scores), best_policy

