from einops import (
    einops,
)

import torch


class AblationDecoderLayer(torch.nn.Module):
    def __init__(self, original_layer, direction):
        """
        Initialize the AblationDecoderLayer.
        :param original_layer: the original layer to be ablated
        :param direction: the direction of the ablation.
        """
        super(AblationDecoderLayer, self).__init__()
        self.original_layer = original_layer
        self.objective_behaviour_dr = direction

    def forward(self, *args, **kwargs):
        """
        Forward method of the AblationDecoderLayer.
        :param args: the arguments to be passed to the original layer
        :param kwargs: the keyword arguments to be passed to the original layer
        :return: the output of the original layer after ablation.
        """
        hidden_states = args[0]
        ablated = self._direction_ablation_hook(
            activation=hidden_states,
            direction=self.objective_behaviour_dr.to(hidden_states.device),
        ).to(hidden_states.device)
        args = (ablated,) + args[1:]
        return self.original_layer.forward(*args, **kwargs)

    @staticmethod
    def _direction_ablation_hook(
        activation: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ablation hook for the AblationDecoderLayer.
        :param activation: the activation to be ablated
        :param direction: the direction of the ablation
        :return: the ablated activation
        """
        proj = (
            einops.einsum(
                activation,
                direction.view(-1, 1),
                "... d_act, d_act single -> ... single",
            )
            * direction
        )
        return activation - proj


class AdditionDecoderLayer(torch.nn.Module):
    def __init__(self, original_layer, direction):
        """
        Initialize the AdditionDecoderLayer.
        :param original_layer: the original layer to be added to
        :param direction: the direction of the addition.
        """
        super(AdditionDecoderLayer, self).__init__()
        self.original_layer = original_layer
        self.objective_behaviour_dir = direction

    def forward(self, *args, **kwargs):
        """
        Forward method of the AdditionDecoderLayer.
        :param args: the arguments to be passed to the original layer
        :param kwargs: the keyword arguments to be passed to the original layer
        :return: the output of the original layer after addition.
        """
        hidden_states = args[0]
        added = self._direction_addition_hook(
            activation=hidden_states,
            direction=self.objective_behaviour_dir.to(hidden_states.device),
        ).to(hidden_states.device)
        args = (added,) + args[1:]
        return self.original_layer.forward(*args, **kwargs)

    @staticmethod
    def _direction_addition_hook(
        activation: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Addition hook for the AdditionDecoderLayer.
        :param activation: the activation to be added to
        :param direction: the direction of the addition
        :return: the added activation
        """
        return activation + direction
