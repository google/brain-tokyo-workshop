import gin
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from solutions.base_solution import BaseSolution
from solutions.torch_modules import SelfAttentionMatrix
from solutions.torch_modules import AttentionNeuronLayer
from solutions.torch_modules import VisionAttentionNeuronLayer


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
        return self.get_params().size

    def _get_action(self, obs):
        raise NotImplementedError()

    def reset(self):
        pass


@gin.configurable
class MLPSolution(BaseTorchSolution):
    """MLP solution."""

    def __init__(self, device, obs_dim, act_dim, hidden_dim, num_hidden_layers):
        super(MLPSolution, self).__init__(device=device)
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Tanh(),
            ])
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_dim),
            nn.Tanh(),
            *hidden_layers,
            nn.Linear(in_features=hidden_dim, out_features=act_dim),
            nn.Tanh(),
        ).to(self.device)
        self.modules_to_learn.append(self.net)
        print('device={}, #params={}'.format(
            self.device, self.get_num_params()))

    def _get_action(self, obs):
        x = torch.from_numpy(obs.copy()).float().to(self.device)
        return self.net(x).cpu().numpy()


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
        print('num_patches={}'.format(self.num_patches))
        self.attention = SelfAttentionMatrix(
            dim_in=3 * self.patch_size ** 2,
            msg_dim=query_dim,
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

        print('num_params={}'.format(self.get_num_params()))

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
        attention_matrix = self.attention(flattened_patches, flattened_patches)
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


@gin.configurable
class PIFCSolution(BaseTorchSolution):
    """Permutation invariant solution."""

    def __init__(self,
                 device,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_em_dim,
                 num_hidden_layers=1,
                 pi_layer_bias=True,
                 pi_layer_scale=True):
        super(PIFCSolution, self).__init__(device=device)
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.prev_act = torch.zeros(1, self.act_dim)

        self.pi_layer = AttentionNeuronLayer(
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim,
            bias=pi_layer_bias,
            scale=pi_layer_scale,
        )
        self.modules_to_learn.append(self.pi_layer)

        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Tanh(),
            ])
        self.net = nn.Sequential(
            *hidden_layers,
            nn.Linear(in_features=hidden_dim, out_features=act_dim),
            nn.Tanh(),
        )
        self.modules_to_learn.append(self.net)

        print('#params={}'.format(self.get_num_params()))

    def _get_action(self, obs):
        x = self.pi_layer(obs=obs, prev_act=self.prev_act)
        self.prev_act = self.net(x.T)
        return self.prev_act.squeeze(0).cpu().numpy()

    def reset(self):
        self.prev_act = torch.zeros(1, self.act_dim)
        self.pi_layer.reset()


@gin.configurable
class PIAttentionAgent(BaseTorchSolution):
    """AttentionNeuron + AttentionAgent."""

    def __init__(self,
                 device,
                 act_dim,
                 msg_dim,
                 pos_em_dim,
                 patch_size=6,
                 stack_k=4,
                 aa_image_size=32,
                 aa_query_dim=4,
                 aa_hidden_dim=16,
                 aa_top_k=10):
        super(PIAttentionAgent, self).__init__(device)
        self.alpha = 0.
        self.attended_patch_ix = None
        self.act_dim = act_dim
        self.prev_act = torch.zeros(1, self.act_dim)
        self.hidden_dim = aa_image_size**2
        self.msg_dim = msg_dim
        self.prev_hidden = torch.zeros(self.hidden_dim, self.msg_dim)

        self.vision_pi_layer = VisionAttentionNeuronLayer(
            act_dim=act_dim,
            hidden_dim=aa_image_size**2,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim,
            patch_size=patch_size,
            stack_k=stack_k,
        )
        self.modules_to_learn.append(self.vision_pi_layer)

        self.top_k = aa_top_k
        self.patch_centers = torch.div(torch.tensor(
            [[i, j] for i in range(aa_image_size) for j in range(aa_image_size)]
        ).float(), aa_image_size)
        self.attention = SelfAttentionMatrix(
            dim_in=self.msg_dim,
            msg_dim=aa_query_dim,
            scale=False,
        )
        self.modules_to_learn.append(self.attention)

        self.hx = None
        self.lstm_hidden_dim = aa_hidden_dim
        self.lstm = nn.LSTMCell(
            input_size=aa_top_k * 2,
            hidden_size=aa_hidden_dim,
        )
        self.modules_to_learn.append(self.lstm)

        self.output_fc = nn.Sequential(
            nn.Linear(in_features=aa_hidden_dim, out_features=act_dim),
            nn.Tanh(),
        )
        self.modules_to_learn.append(self.output_fc)

        self.mixing_fc = nn.Sequential(
            nn.Linear(
                in_features=aa_hidden_dim + self.act_dim,
                out_features=aa_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(in_features=aa_hidden_dim, out_features=1),
            nn.Sigmoid(),
        )
        self.modules_to_learn.append(self.mixing_fc)

        print('#params={}'.format(self.get_params().size))

    def _get_action(self, obs):
        # Uncomment to confirm the agent is receiving shuffled obs.
        # import cv2
        # viz_obs = obs[0]
        # cv2.imshow('confirm', cv2.resize(viz_obs, (400, 400)))
        # cv2.waitKey(1)

        x = self.vision_pi_layer(obs=obs, prev_act=self.prev_act)
        self.attended_patch_ix = (
            self.vision_pi_layer.attention.mostly_attended_entries
        )
        x = (1 - self.alpha) * x + self.alpha * self.prev_hidden
        self.prev_hidden = x

        attention_matrix = self.attention(data_q=x, data_k=x)
        patch_importance_matrix = torch.softmax(attention_matrix, dim=-1)
        patch_importance = patch_importance_matrix.sum(dim=0)

        # Extract top k important patches
        ix = torch.argsort(patch_importance, descending=True)
        top_k_ix = ix[:self.top_k]
        centers = self.patch_centers[top_k_ix]
        centers = centers.flatten(0, -1)

        if self.hx is None:
            self.hx = (
                torch.zeros(1, self.lstm_hidden_dim),
                torch.zeros(1, self.lstm_hidden_dim),
            )
        self.hx = self.lstm(centers.unsqueeze(0), self.hx)
        output = self.output_fc(self.hx[0])
        self.prev_act = output

        self.alpha = self.mixing_fc(
            torch.cat([self.hx[0], self.prev_act], dim=-1).squeeze(0))

        return output.squeeze(0).cpu().numpy()

    def reset(self):
        self.alpha = 0.
        self.prev_act = torch.zeros(1, self.act_dim)
        self.prev_hidden = torch.zeros(self.hidden_dim, self.msg_dim)
        self.hx = None


@gin.configurable
class PuzzlePongSolution(BaseTorchSolution):
    """AttentionNeuron + Convnet."""

    def __init__(self,
                 device,
                 act_dim,
                 msg_dim,
                 pos_em_dim,
                 patch_size=6,
                 stack_k=4,
                 feat_dim=20):
        super(PuzzlePongSolution, self).__init__(device)
        self.act_dim = act_dim
        self.prev_action = None
        self.feat_dim = feat_dim
        self.msg_dim = msg_dim

        self.vision_pi_layer = VisionAttentionNeuronLayer(
            act_dim=act_dim,
            hidden_dim=feat_dim**2,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim,
            patch_size=patch_size,
            stack_k=stack_k,
            with_learnable_ln_params=True,
            stack_dim_first=True,
        )
        self.modules_to_learn.append(self.vision_pi_layer)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=msg_dim,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=act_dim),
        )
        self.modules_to_learn.append(self.cnn)

        print('#params={}'.format(self.get_params().size))

    def _get_action(self, obs):
        if self.prev_action is None:
            self.prev_action = torch.zeros(1, self.act_dim)
            self.prev_action[:, 3] = 1
        x = self.vision_pi_layer(obs=obs, prev_act=self.prev_action)
        assert x.shape == (self.feat_dim**2, self.msg_dim)

        # Reshape to input to convnet.
        x = x.reshape(self.feat_dim, self.feat_dim, self.msg_dim).unsqueeze(0)
        x = torch.relu(x.permute(0, 3, 1, 2))

        action = self.cnn(x)
        assert action.shape == (1, self.act_dim)
        action = torch.argmax(action, dim=-1)
        self.prev_action = torch.zeros(1, self.act_dim)
        self.prev_action[:, action[0]] = 1

        return action

    def reset(self):
        self.prev_action = None
