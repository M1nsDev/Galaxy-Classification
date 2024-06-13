from pathlib import Path
import torch


def save_model_params(model: torch.nn.Module, destination: Path) -> None:
    torch.save(model.state_dict(), destination)

def load_model_params(model: torch.nn.Module, checkpoint_file: Path) -> None:
    model.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))