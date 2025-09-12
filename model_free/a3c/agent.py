import torch
import threading
from agent import A3C
class A3CAgent:
    def __init__(
        self,
        input_dims,
        num_actions,
        learning_rate=1e-4,
        n_steps=6,
        num_threads=16,
        gamma=0.99,
        save_model_interval=100,
        checkpoint_dir="checkpoints",
        model_base_name="a3c_model",
    ):
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.num_threads = num_threads
        self.gamma = gamma

        self.save_model_interval = save_model_interval
        self.checkpoint_dir = checkpoint_dir
        self.model_base_name = model_base_name

        self.device = torch.device("cpu")
        self.global_model = A3C(input_dims[0], num_actions).to(self.device)
        self.global_model.share_memory()  # Allow sharing across threads
        self.optimizer = torch.optim.RMSprop(
            self.global_model.parameters(), lr=learning_rate
        )
        self.global_counter = 0
        self.lock = threading.Lock()
        self.episode_rewards = []
        self.episode_counter = 0

    def train(self, env_fn, Tmax):
        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=self._worker, args=(env_fn, Tmax))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def _worker(self, env_fn, Tmax):
        env = env_fn()
        local_model = A3C(self.input_dims[0], self.num_actions).to(self.device)
        state, _ = env.reset()
        done = False
        t = 1
        hx, cx = None, None
        current_episode_reward = 0  # Initialize episode reward
        while True:
            # Synchronize local parameters with global
            local_model.load_state_dict(self.global_model.state_dict())
            states, actions, rewards, values, log_probs = [], [], [], [], []
            t_start = t
            hx, cx = None, None  # Reset LSTM state at the start of rollout
            # t - t_start is the number of steps in the current rollout
            while not done and t - t_start < self.n_steps:
                # Prepare input for A3C: (batch, seq_len, c, h, w)
                state_tensor = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                )  # (1, 1, num_stacked_frames, height, width)
                logits, value, (hx, cx) = local_model(state_tensor, hx, cx)
                logits = logits.squeeze(0)  # (1, num_actions) -> (num_actions,)
                value = value.squeeze(0)  # (1,) -> ()
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                log_prob = torch.log_softmax(logits, dim=-1)[0, action]
                next_state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                state = next_state
                t += 1
                current_episode_reward += reward  # Accumulate reward
                with self.lock:
                    self.global_counter += 1
                    if self.global_counter > Tmax:
                        return
            # Bootstrap value
            if done:
                R = 0
            else:
                state_tensor = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                )
                _, value, _ = local_model(state_tensor, hx, cx)
                R = value.squeeze(0).item()
            # Compute n-step returns and accumulate gradients
            policy_loss = 0
            value_loss = 0
            for i in reversed(range(len(rewards))):
                R = rewards[i] + self.gamma * R
                advantage = R - values[i]
                policy_loss += -log_probs[i] * advantage.detach()
                # Ensure both tensors are 1D of shape [1] for mse_loss
                target = torch.tensor([R], device=self.device, dtype=values[i].dtype)
                value_loss += F.mse_loss(values[i].view(-1), target)
                print(f"Value loss: {value_loss.item()}, Policy loss: {policy_loss.item()}")
            # Backprop and async update
            # Compute gradients on local model only
            local_model.zero_grad()
            (policy_loss + value_loss).backward()
            # Thread-safe global model update
            with self.lock:
                self.optimizer.zero_grad()
                # Set all grads to None first
                for global_param in self.global_model.parameters():
                    global_param.grad = None
                # Overwrite with local grads if available, robustly
                global_params = list(self.global_model.parameters())
                local_params = list(local_model.parameters())
                if len(global_params) != len(local_params):
                    print(
                        f"WARNING: global_model and local_model have different number of parameters: {len(global_params)} vs {len(local_params)}"
                    )
                for global_param, local_param in zip(global_params, local_params):
                    if local_param.grad is not None:
                        global_param.grad = local_param.grad.clone()
                    # else: leave as None
                self.optimizer.step()
                self.global_model.zero_grad()
            local_model.zero_grad()
            if done:
                state, _ = env.reset()
                done = False

            with self.lock:
                self.episode_rewards.append(current_episode_reward)
                self.episode_counter += 1
                log_dict = {
                    "episode": self.episode_counter,
                    "episode_reward": current_episode_reward,
                    "total_frames": self.global_counter,
                }
                save_checkpoint = False
                if self.episode_counter % self.save_model_interval == 0:
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f"{self.model_base_name}_ep{self.episode_counter}.pth",
                    )
                    self.save_model(checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
                    save_checkpoint = True
            # wandb.log outside the lock for thread safety
            # wandb.log(log_dict)
            current_episode_reward = 0  # Reset for next episode

    def save_model(self, path):
        torch.save(
            {
                "global_model_state_dict": self.global_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_counter": self.global_counter,
            },
            path,
        )