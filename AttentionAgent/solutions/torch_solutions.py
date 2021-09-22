import gin
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from solutions.base_solution import BaseSolution


torch.set_num_threads(1)


class BaseTorchSolution(BaseSolution):
    """Basic torch solution."""

    def __init__(self, device):
        self.modules_to_learn = []
        self.device = torch.device(device)

    def get_action(self, obs):
        with torch.no_grad():
            return self._get_action(obs)

    def get_params(self):
        params = []
        with torch.no_grad():
            for layer in self.modules_to_learn:
                for p in layer.parameters():
                    params.append(p.cpu().numpy().ravel())
        return np.concatenate(params)

    def set_params(self, params):
        assert isinstance(params, np.ndarray)
        ss = 0
        for layer in self.modules_to_learn:
            for p in layer.parameters():
                ee = ss + np.prod(p.shape)
                p.data = torch.from_numpy(
                    params[ss:ee].reshape(p.shape)
                ).float().to(self.device)
                ss = ee
        assert ss == params.size

    def save(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def load(self, filename):
        with np.load(filename) as data:
            params = data['params']
            self.set_params(params)

    def get_num_params(self):
        num_params = 0
        for layer in self.modules_to_learn:
            for p in layer.parameters():
                num_params += np.prod(p.shape)
        return num_params

    def _get_action(self, obs):
        raise NotImplementedError()

    def reset(self):
        pass


@gin.configurable
class MLPSolution(BaseTorchSolution):
    """MLP solution."""

    def __init__(self, device, obs_dim, act_dim, hidden_dim, num_hidden_layers):
        super(MLPSolution, self).__init__(device=device)
        hiddens = []
        for _ in range(num_hidden_layers):
            hiddens.extend([
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Tanh(),
            ])
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_dim),
            nn.Tanh(),
            *hiddens,
            nn.Linear(in_features=hidden_dim, out_features=act_dim),
            nn.Tanh(),
        ).to(self.device)
        self.modules_to_learn.append(self.net)
        print('device={}, #params={}'.format(
            self.device, self.get_num_params()))

    def _get_action(self, obs):
        x = torch.from_numpy(obs.copy()).float().to(self.device)
        return self.net(x).cpu().numpy()


class SelfAttention(nn.Module):
    """A simple self-attention solution."""

    def __init__(self, data_dim, dim_q):
        super(SelfAttention, self).__init__()
        self._layers = []

        self._fc_q = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_q)
        self._fc_k = nn.Linear(data_dim, dim_q)
        self._layers.append(self._fc_k)

    def forward(self, input_data):
        # Expect input_data to be of shape (b, t, k).
        b, t, k = input_data.size()

        # Linear transforms.
        queries = self._fc_q(input=input_data)  # (b, t, q)
        keys = self._fc_k(input=input_data)  # (b, t, q)

        # Attention matrix.
        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b, t, t)
        scaled_dot = torch.div(dot, torch.sqrt(torch.tensor(k).float()))
        return scaled_dot


@gin.configurable
class AttentionAgent(BaseTorchSolution):
    """Attention Agent solution."""

    def __init__(self,
                 device,
                 image_size=96,
                 patch_size=7,
                 patch_stride=4,
                 query_dim=4,
                 hidden_dim=16,
                 top_k=10):
        super(AttentionAgent, self).__init__(device=device)
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        n = int((image_size - patch_size) / patch_stride + 1)
        offset = self.patch_size // 2
        patch_centers = []
        for i in range(n):
            patch_center_row = offset + i * patch_stride
            for j in range(n):
                patch_center_col = offset + j * patch_stride
                patch_centers.append([patch_center_row, patch_center_col])
        self.patch_centers = torch.tensor(patch_centers).float()

        self.num_patches = n ** 2
        print('num_patches = {}'.format(self.num_patches))
        self.attention = SelfAttention(
            data_dim=3 * self.patch_size ** 2,
            dim_q=query_dim,
        )
        self.modules_to_learn.append(self.attention)

        self.hx = None
        self.lstm = nn.LSTMCell(
            input_size=self.top_k * 2,
            hidden_size=hidden_dim,
        )
        self.modules_to_learn.append(self.lstm)

        self.output_fc = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=3),
            nn.Tanh(),
        )
        self.modules_to_learn.append(self.output_fc)

        print('num_params={}'.format(self.get_params().size))

    def _get_action(self, obs):
        # ob.shape = (h, w, c)
        ob = self.transform(obs).permute(1, 2, 0)
        h, w, c = ob.size()
        patches = ob.unfold(
            0, self.patch_size, self.patch_stride).permute(0, 3, 1, 2)
        patches = patches.unfold(
            2, self.patch_size, self.patch_stride).permute(0, 2, 1, 4, 3)
        patches = patches.reshape((-1, self.patch_size, self.patch_size, c))

        # flattened_patches.shape = (1, n, p * p * c)
        flattened_patches = patches.reshape(
            (1, -1, c * self.patch_size ** 2))
        # attention_matrix.shape = (1, n, n)
        attention_matrix = self.attention(flattened_patches)
        # patch_importance_matrix.shape = (n, n)
        patch_importance_matrix = torch.softmax(
            attention_matrix.squeeze(), dim=-1)
        # patch_importance.shape = (n,)
        patch_importance = patch_importance_matrix.sum(dim=0)
        # extract top k important patches
        ix = torch.argsort(patch_importance, descending=True)
        top_k_ix = ix[:self.top_k]

        centers = self.patch_centers[top_k_ix]
        centers = centers.flatten(0, -1)
        centers = centers / self.image_size

        if self.hx is None:
            self.hx = (
                torch.zeros(1, self.hidden_dim),
                torch.zeros(1, self.hidden_dim),
            )
        self.hx = self.lstm(centers.unsqueeze(0), self.hx)
        output = self.output_fc(self.hx[0]).squeeze(0)
        return output.cpu().numpy()

    def reset(self):
        self.hx = None
