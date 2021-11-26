import os
import util
import numpy as np

import torch
import torch.nn as nn
from solutions.torch_modules import AttentionNeuronLayer


ACT_DIM = 8


class PIStudent(nn.Module):
    """Permutation invariant student policy."""

    def __init__(self, act_dim, hidden_dim, msg_dim, pos_em_dim):
        super(PIStudent, self).__init__()

        self.attention_neuron = AttentionNeuronLayer(
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim,
        )
        self.policy_net = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=act_dim),
            nn.Tanh(),
        )

    def forward(self, obs, prev_act):
        msg = self.attention_neuron(obs=obs, prev_act=prev_act)
        return self.policy_net(msg.T)


def load_data(data_file):
    with np.load(data_file, 'r') as data:
        trajectories = data['data']
    return trajectories


def sample_batch_data(data, batch_size, seed=0):
    num_rollouts, traj_len, data_dim = data.shape
    rnd = np.random.RandomState(seed=seed)
    while True:
        rollout_ix = rnd.choice(num_rollouts, batch_size, replace=False)
        yield data[rollout_ix]


def save_model(model, i):
    params = []
    for p in model.parameters():
        params.append(p.cpu().data.numpy().ravel())
    params = np.concatenate(params, axis=0)
    np.savez(
        'pretrained/ant_pi/bc_train_iter{0:06d}.npz'.format(i), params=params)


def main(log_dir):
    logger = util.create_logger(name='bc_training', log_dir='pretrained/ant_pi')

    device = torch.device('cuda:0')
    policy = PIStudent(
        act_dim=ACT_DIM,
        msg_dim=32,
        pos_em_dim=8,
        hidden_dim=32,
    ).to(device)

    batch_size = 8
    data = load_data(os.path.join(log_dir, 'data.npz'))
    batches = sample_batch_data(data, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    max_iter = 1000000
    noise_sd = 0.1
    for i in range(max_iter):
        batch_data = torch.from_numpy(next(batches)).float().to(device)
        optimizer.zero_grad()

        # This is only to show how BC training works, it is inefficient.
        seq_len = batch_data.shape[1]
        losses = []
        for traj in batch_data:
            pred_act = []
            policy.attention_neuron.reset()  # Reset AttentionNeuron's hx.
            for t in range(seq_len):
                prev_act, obs = traj[t][:ACT_DIM], traj[t][ACT_DIM:-ACT_DIM]
                prev_act = prev_act + torch.randn(ACT_DIM).to(device) * noise_sd
                pred_act.append(policy(obs, prev_act))
            pred_act = torch.vstack(pred_act)
            act = traj[:, -ACT_DIM:]
            losses.append(criterion(input=pred_act, target=act))
        loss = sum(losses) / batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=.1)
        optimizer.step()

        logger.info('iter={}, loss={}'.format(i, loss.item()))
        if i % 1000 == 0:
            save_model(policy, i)

    save_model(policy, max_iter)


if __name__ == '__main__':
    model_dir = 'pretrained/ant_mlp'
    main(model_dir)
