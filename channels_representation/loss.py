import torch

class TargetWeights():
    """
    Weighted function.
    """
    def __init__(self, global_max_value: float = 1.0, percentage: float = 0.01):
        """
        Args:
            global_max_value: Maximum value for the global intensity.
            percentage: Percentage of the maximum value to use as a threshold for weighting.
        """
        self.max_global_value = global_max_value
        self.percentage = percentage
        self.threshold = global_max_value * percentage

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: Tensor of shape (B, N) - target values.

        Returns:
            Weighted MSE loss.
        """
        # Calculate the weight tensor
        # Calculate the threshold as a tensor on the same device
        threshold = torch.tensor(self.threshold, device=target.device)
        weight = 1.0 / torch.maximum(threshold, target)
        return weight