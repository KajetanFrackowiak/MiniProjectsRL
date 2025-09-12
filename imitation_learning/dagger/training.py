import os
import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(
        self, env, env_name, seed, expert, student, optimizer, loss_fn, device
    ):
        self.env = env
        self.env_name = env_name
        self.seed = seed
        self.expert = expert
        self.student = student
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.student.to(self.device)

    def collect_data(self, rollouts=10, beta=0.0):
        states, expert_actions = [], []

        for _ in range(rollouts):
            state, _ = self.env.reset()
            done = False
            while not done:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    student_logits = self.student(state_tensor)
                    student_action = student_logits.argmax(dim=1).item()

                    expert_action, _ = self.expert.predict(state, deterministic=True)

                    states.append(state_tensor.squeeze(0))
                    expert_actions.append(expert_action)

                # mixture policy: with prob beta use expert action, otherwise student action
                if torch.rand(1).item() < beta:
                    action_to_step = expert_action
                else:
                    action_to_step = student_action

                state, reward, terminated, truncated, _ = self.env.step(action_to_step)
                done = terminated or truncated

        return states, expert_actions

    def train_student(self, states, expert_actions):
        self.student.train()
        self.optimizer.zero_grad()
        if len(states) == 0:
            return None
        states = torch.stack(states).to(self.device)

        expert_actions = np.array(expert_actions).ravel()
        targets = torch.as_tensor(expert_actions, dtype=torch.long, device=self.device)
        logits = self.student(states)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self, iterations=10, rollouts_per_iter=10, p=0.9):
        agg_states, agg_actions = [], []
        losses = []
        for i in tqdm(range(iterations)):
            beta = p**i
            states, actions = self.collect_data(rollouts=rollouts_per_iter, beta=beta)
            agg_states.extend(states)
            agg_actions.extend(actions)
            loss = self.train_student(agg_states, agg_actions)
            losses.append(loss)
            tqdm.write(f"Iteration {i + 1}/{iterations}, Loss: {loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            {
                "student_state_dict": self.student.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "losses": losses,
            },
            f"checkpoints/student_env_{self.env_name}_iter_{iterations}_seed_{self.seed}.pth",
        )
        return agg_states, agg_actions, losses
