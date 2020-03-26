import abc
import gin
import numpy as np
import os
import solutions.abc_solution
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class BaseTorchSolution(solutions.abc_solution.BaseSolution):
    """Base class for all Torch solutions."""

    def __init__(self):
        self._layers = []

    def get_output(self, inputs, update_filter=False):
        torch.set_num_threads(1)
        with torch.no_grad():
            return self._get_output(inputs, update_filter)

    @abc.abstractmethod
    def _get_output(self, inputs, update_filter):
        raise NotImplementedError()

    def get_params(self):
        params = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)

    def set_params(self, params):
        offset = 0
        for i, layer in enumerate(self._layers):
            weights_to_set = {}
            weight_dict = layer.state_dict()
            for k in sorted(weight_dict.keys()):
                weight = weight_dict[k].numpy()
                weight_size = weight.size
                weights_to_set[k] = torch.from_numpy(
                    params[offset:(offset + weight_size)].reshape(weight.shape))
                offset += weight_size
            self._layers[i].load_state_dict(state_dict=weights_to_set)

    def get_params_from_layer(self, layer_index):
        params = []
        layer = self._layers[layer_index]
        weight_dict = layer.state_dict()
        for k in sorted(weight_dict.keys()):
            params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)

    def set_params_to_layer(self, params, layer_index):
        weights_to_set = {}
        weight_dict = self._layers[layer_index].state_dict()
        offset = 0
        for k in sorted(weight_dict.keys()):
            weight = weight_dict[k].numpy()
            weight_size = weight.size
            weights_to_set[k] = torch.from_numpy(
                params[offset:(offset + weight_size)].reshape(weight.shape))
            offset += weight_size
        self._layers[layer_index].load_state_dict(state_dict=weights_to_set)

    def get_num_params_per_layer(self):
        num_params_per_layer = []
        for layer in self._layers:
            weight_dict = layer.state_dict()
            num_params = 0
            for k in sorted(weight_dict.keys()):
                weights = weight_dict[k].numpy()
                num_params += weights.size
            num_params_per_layer.append(num_params)
        return num_params_per_layer

    def _save_to_file(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def save(self, log_dir, iter_count, best_so_far):
        filename = os.path.join(log_dir, 'model_{}.npz'.format(iter_count))
        self._save_to_file(filename=filename)
        if best_so_far:
            filename = os.path.join(log_dir, 'best_model.npz')
            self._save_to_file(filename=filename)

    def load(self, filename):
        with np.load(filename) as data:
            params = data['params']
            self.set_params(params)

    def reset(self):
        raise NotImplementedError()

    @property
    def layers(self):
        return self._layers


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

    @property
    def layers(self):
        return self._layers


class FCStack(nn.Module):
    """Fully connected layers."""

    def __init__(self, input_dim, num_units, activation, output_dim):
        super(FCStack, self).__init__()
        self._activation = activation
        self._layers = []
        dim_in = input_dim
        for i, n in enumerate(num_units):
            layer = nn.Linear(dim_in, n)
            # layer.weight.data.fill_(0.0)
            # layer.bias.data.fill_(0.0)
            self._layers.append(layer)
            setattr(self, '_fc{}'.format(i + 1), layer)
            dim_in = n
        output_layer = nn.Linear(dim_in, output_dim)
        # output_layer.weight.data.fill_(0.0)
        # output_layer.bias.data.fill_(0.0)
        self._layers.append(output_layer)

    @property
    def layers(self):
        return self._layers

    def forward(self, input_data):
        x_input = input_data
        for layer in self._layers[:-1]:
            x_output = layer(x_input)
            if self._activation == 'tanh':
                x_input = torch.tanh(x_output)
            elif self._activation == 'elu':
                x_input = F.elu(x_output)
            else:
                x_input = F.relu(x_output)
        x_output = self._layers[-1](x_input)
        return x_output


class LSTMStack(nn.Module):
    """LSTM layers."""

    def __init__(self, input_dim, num_units, output_dim):
        super(LSTMStack, self).__init__()
        self._layers = []
        self._hidden_layers = len(num_units) if len(num_units) else 1
        self._hidden_size = num_units[0] if len(num_units) else output_dim
        self._hidden = (
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
        )
        if len(num_units):
            self._lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self._hidden_size,
                num_layers=self._hidden_layers,
            )
            self._layers.append(self._lstm)
            fc = nn.Linear(
                in_features=self._hidden_size,
                out_features=output_dim,
            )
            self._layers.append(fc)
        else:
            self._lstm = nn.LSTMCell(
                input_size=input_dim,
                hidden_size=self._hidden_size,
            )
            self._layers.append(self._lstm)

    @property
    def layers(self):
        return self._layers

    def forward(self, input_data):
        x_input = input_data
        x_output, self._hidden = self._layers[0](
            x_input.view(1, 1, -1), self._hidden)
        x_output = torch.flatten(x_output, start_dim=0, end_dim=-1)
        if len(self._layers) > 1:
            x_output = self._layers[-1](x_output)
        return x_output

    def reset(self):
        self._hidden = (
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
            torch.zeros((self._hidden_layers, 1, self._hidden_size)),
        )


@gin.configurable
class MLPSolution(BaseTorchSolution):
    """Multi-layer perception."""

    def __init__(self,
                 input_dim,
                 num_hiddens,
                 activation,
                 output_dim,
                 output_activation,
                 use_lstm,
                 l2_coefficient):
        super(MLPSolution, self).__init__()
        self._use_lstm = use_lstm
        self._output_dim = output_dim
        self._output_activation = output_activation
        if 'roulette' in self._output_activation:
            assert self._output_dim == 1
            self._n_grid = int(self._output_activation.split('_')[-1])
            self._theta_per_grid = 2 * np.pi / self._n_grid
        self._l2_coefficient = abs(l2_coefficient)
        if self._use_lstm:
            self._fc_stack = LSTMStack(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=num_hiddens,
            )
        else:
            self._fc_stack = FCStack(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=num_hiddens,
                activation=activation,
            )
        self._layers = self._fc_stack.layers
        print('Number of parameters: {}'.format(
            self.get_num_params_per_layer()))

    def _get_output(self, inputs, update_filter=False):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs).float()
        fc_output = self._fc_stack(inputs)

        if self._output_activation == 'tanh':
            output = torch.tanh(fc_output).squeeze().numpy()
        elif self._output_activation == 'softmax':
            output = F.softmax(fc_output, dim=-1).squeeze().numpy()
        else:
            output = fc_output.squeeze().numpy()

        return output

    def reset(self):
        if hasattr(self._fc_stack, 'reset'):
            self._fc_stack.reset()
            print('hidden reset.')


@gin.configurable
class VisionTaskSolution(BaseTorchSolution):
    """A general solution for vision based tasks."""

    def __init__(self,
                 image_size,
                 query_dim,
                 output_dim,
                 output_activation,
                 num_hiddens,
                 l2_coefficient,
                 patch_size,
                 patch_stride,
                 top_k,
                 data_dim,
                 activation,
                 normalize_positions=False,
                 use_lstm_controller=False,
                 show_overplot=False):
        super(VisionTaskSolution, self).__init__()
        self._image_size = image_size
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._top_k = top_k
        self._l2_coefficient = l2_coefficient
        self._show_overplot = show_overplot
        self._normalize_positions = normalize_positions
        self._screen_dir = None
        self._img_ix = 1
        self._raw_importances = []

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        n = int((image_size - patch_size) / patch_stride + 1)
        offset = self._patch_size // 2
        patch_centers = []
        for i in range(n):
            patch_center_row = offset + i * patch_stride
            for j in range(n):
                patch_center_col = offset + j * patch_stride
                patch_centers.append([patch_center_row, patch_center_col])
        self._patch_centers = torch.tensor(patch_centers).float()

        num_patches = n ** 2
        print('num_patches = {}'.format(num_patches))
        self._attention = SelfAttention(
            data_dim=data_dim * self._patch_size ** 2,
            dim_q=query_dim,
        )
        self._layers.extend(self._attention.layers)

        self._mlp_solution = MLPSolution(
            input_dim=self._top_k * 2,
            num_hiddens=num_hiddens,
            activation=activation,
            output_dim=output_dim,
            output_activation=output_activation,
            l2_coefficient=l2_coefficient,
            use_lstm=use_lstm_controller,
        )
        self._layers.extend(self._mlp_solution.layers)

        print('Number of parameters: {}'.format(
            self.get_num_params_per_layer()))

    def _get_output(self, inputs, update_filter):

        # ob.shape = (h, w, c)
        ob = self._transform(inputs).permute(1, 2, 0)
        # print(ob.shape)
        h, w, c = ob.size()
        patches = ob.unfold(
            0, self._patch_size, self._patch_stride).permute(0, 3, 1, 2)
        patches = patches.unfold(
            2, self._patch_size, self._patch_stride).permute(0, 2, 1, 4, 3)
        patches = patches.reshape((-1, self._patch_size, self._patch_size, c))

        # flattened_patches.shape = (1, n, p * p * c)
        flattened_patches = patches.reshape(
            (1, -1, c * self._patch_size ** 2))
        # attention_matrix.shape = (1, n, n)
        attention_matrix = self._attention(flattened_patches)
        # patch_importance_matrix.shape = (n, n)
        patch_importance_matrix = torch.softmax(
            attention_matrix.squeeze(), dim=-1)
        # patch_importance.shape = (n,)
        patch_importance = patch_importance_matrix.sum(dim=0)
        # extract top k important patches
        ix = torch.argsort(patch_importance, descending=True)
        top_k_ix = ix[:self._top_k]

        centers = self._patch_centers[top_k_ix]

        # Overplot.
        if self._show_overplot:
            task_image = ob.numpy().copy()
            patch_importance_copy = patch_importance.numpy().copy()

            import cv2
            if self._screen_dir is not None:
                # Save the original screen.
                img_filepath = os.path.join(
                    self._screen_dir, 'orig_{0:04d}.png'.format(self._img_ix))
                cv2.imwrite(img_filepath, inputs[:, :, ::-1])
                # Save the scaled screen.
                img_filepath = os.path.join(
                    self._screen_dir, 'scaled_{0:04d}.png'.format(self._img_ix))
                cv2.imwrite(
                    img_filepath,
                    (task_image * 255).astype(np.uint8)[:, :, ::-1]
                )
                # Save importance vectors.
                dd = {
                    'step': self._img_ix,
                    'importance': patch_importance_copy.tolist(),
                }
                self._raw_importances.append(dd)
                import pandas as pd
                if self._img_ix % 20 == 0:
                    csv_path = os.path.join(self._screen_dir, 'importances.csv')
                    pd.DataFrame(self._raw_importances).to_csv(
                        csv_path, index=False
                    )

            white_patch = np.ones(
                (self._patch_size, self._patch_size, 3))
            half_patch_size = self._patch_size // 2
            for i, center in enumerate(centers):
                row_ss = int(center[0]) - half_patch_size
                row_ee = int(center[0]) + half_patch_size + 1
                col_ss = int(center[1]) - half_patch_size
                col_ee = int(center[1]) + half_patch_size + 1
                ratio = 1.0 * i / self._top_k
                task_image[row_ss:row_ee, col_ss:col_ee] = (
                        ratio * task_image[row_ss:row_ee, col_ss:col_ee] +
                        (1 - ratio) * white_patch)
            task_image = cv2.resize(
                task_image, (task_image.shape[0] * 5, task_image.shape[1] * 5))
            cv2.imshow('Overplotting', task_image[:, :, [2, 1, 0]])
            cv2.waitKey(1)

            if self._screen_dir is not None:
                # Save the scaled screen.
                img_filepath = os.path.join(
                    self._screen_dir, 'att_{0:04d}.png'.format(self._img_ix))
                cv2.imwrite(
                    img_filepath,
                    (task_image * 255).astype(np.uint8)[:, :, ::-1]
                )

            self._img_ix += 1

        centers = centers.flatten(0, -1)
        if self._normalize_positions:
            centers = centers / self._image_size

        return self._mlp_solution.get_output(centers)

    def reset(self):
        self._selected_patch_centers = []
        self._value_network_input_images = []
        self._accumulated_gradients = None
        self._mlp_solution.reset()
        self._img_ix = 1
        self._raw_importances = []

    def set_log_dir(self, folder):
        self._screen_dir = folder
        if not os.path.exists(self._screen_dir):
            os.makedirs(self._screen_dir)
