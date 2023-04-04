import torch as th
import numpy as np

from torch.nn import functional as F
from stable_baselines3 import DQN

from torch.nn import functional as F

class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            ### YOUR CODE HERE
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Do not backpropagate gradient to the target network
            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Decouple action selection from value estimation
                # Compute q-values for the next observation using the online q net
                next_q_values_online = self.q_net(replay_data.next_observations) 
                # Select action with online network
                next_actions_online = next_q_values_online.argmax(dim=1).view(batch_size, 1)
                self.logger.debug(f'next_q_value_online.shape={next_q_values_online.shape}')
                self.logger.debug(f'next_actions_online={next_actions_online}')

                # Estimate the q-values for the selected actions using target q network
                next_q_values = next_q_values.gather(1, next_actions_online)
                self.logger.debug(next_q_values.shape)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Check the shape
            self.logger.debug(f'current_q_values.shape={current_q_values.shape}, target_q_values.shape={target_q_values.shape}')
            assert current_q_values.shape == target_q_values.shape

            # Compute loss (L2 or Huber loss)
            loss =  F.smooth_l1_loss(current_q_values, target_q_values)

            ### END OF YOUR CODE
            losses.append(loss.item())

            # Optimize the q-network
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))