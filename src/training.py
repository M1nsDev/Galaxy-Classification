from functools import partial
from typing import Any, Optional

from pathlib import Path
import pandas as pd

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader



class History:
    def __init__(self) -> None:
        self._history: dict[str, list[float]] = dict()
        self._logged: bool = False

    def __getitem__(self, index: str) -> list[float]:
        return self._history[index]

    def log(self, **kwargs) -> None:
        self._logged = True
        for key, value in kwargs.items():
            if key not in self._history:
                self._history[key] = []
            self._history[key].append(value)

    def has_logged(self) -> bool:
        return self._logged
    
    def to_csv(self, file_path):
        if not self._logged:
            raise ValueError("No data has been logged yet.")
        df = pd.DataFrame(self._history)
        df.to_csv(file_path, index=False)


def train_classifier(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_test: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float = 1e-3,
    lr_scheduler_cls: Optional[type[torch.optim.lr_scheduler.LRScheduler]] = None,
    lr_scheduler_args: Optional[dict[str, Any]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> History:
    data_loader = partial(
        DataLoader,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dl_train = data_loader(ds_train, shuffle=True)
    dl_test = data_loader(ds_test, shuffle=False)



    model_saved_params = Path("./saved/best-model-params.pt")
    if model_saved_params.is_file():
        model.load_state_dict(torch.load(model_saved_params))



    device = next(model.parameters()).device
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = None
    if lr_scheduler_cls is not None and lr_scheduler_args is not None:
        lr_scheduler = lr_scheduler_cls(optimizer=optimizer, **lr_scheduler_args)

    history = History()


    


    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct_predicted = 0
        num_samples = 0

        for samples, labels in tqdm(dl_train, desc="Training", unit="batches"):
            samples = samples.to(device)
            labels = labels.to(device)

            model.zero_grad()

            predictions = model(samples)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted_labels = torch.argmax(predictions, dim=-1)
            correct_predicted += (predicted_labels == labels).sum().item()
            num_samples += len(samples)

        train_loss = running_loss / len(dl_train)
        train_accuracy = correct_predicted / num_samples

        model.eval()

        running_loss = 0.0
        correct_predicted = 0
        num_samples = 0

        with torch.no_grad():
            for samples, labels in tqdm(dl_test, desc="Evaluation", unit="batches"):
                samples = samples.to(device)
                labels = labels.to(device)

                predictions = model(samples)
                loss = loss_fn(predictions, labels)

                running_loss += loss.item()
                predicted_labels = torch.argmax(predictions, dim=-1)
                correct_predicted += (predicted_labels == labels).sum().item()
                num_samples += len(samples)

        test_loss = running_loss / len(dl_test)
        test_accuracy = correct_predicted / num_samples

        if history.has_logged():
            if test_loss < max(history["test_loss"]):
                torch.save(model.state_dict(), "./saved/best-model-params.pt")

        history.log(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
        )

        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss: {train_loss:.04f} ; train_accuracy: {train_accuracy:.04f} ; "
            f"test_loss: {test_loss:.04f} ; test_accuracy: {test_accuracy:.04f}"
        )

        if lr_scheduler is not None:
            lr_scheduler.step()
            print(f"Set learning rate to {lr_scheduler.get_last_lr()[0]:.4E}")

    return history