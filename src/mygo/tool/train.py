"""Train MyGo models."""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from mygo.agent import MLBot
from mygo.dataset import MCTSDataset
from mygo.encoder import OnePlaneEncoder
from mygo.game import Game
from mygo.model import TinyModel

plt.style.use("dark_background")


class ModelTrainer:
    @staticmethod
    def pretty_time(s):
        if s < 0.1:
            ms = s * 1000
            return f"{ms:.1f}ms"
        elif s < 60:
            return f"{s:.1f}s"
        elif s < 3600:
            m = s / 60
            return f"{m:.1f}m"
        else:
            h = s / 3600
            return f"{h:.1f}h"

    @staticmethod
    def default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def transform(data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(ModelTrainer.default_device())
        elif isinstance(data, (int, tuple, list)):
            return torch.tensor(data, device=ModelTrainer.default_device())

    def __init__(
        self,
        device=None,
        board_size=9,  # TODO: -> 19
        encoder=None,
        model_name="tiny",
        model=None,
        root=None,
        max_iters=50_000,
        eval_interval=400,
        eval_iters=200,
        train_data=None,
        test_data=None,
    ):
        # Pytorch settings
        self.device = self.default_device() if device is None else device

        # Game settings
        self.komi = 0.0
        self.board_size = board_size

        # Data settings
        self.encoder = OnePlaneEncoder(self.board_size) if encoder is None else encoder
        # n_planes = encoder.plane_count
        self.n_train_games = 9
        self.n_test_games = 1

        # Model setup
        self.model_name = model_name
        if model is None:
            model = TinyModel(self.board_size)
        self.model = model.to(self.device)
        self.model_info(self.model)

        # File settings
        self.root = Path("data" if root is None else root)
        self.data_root = self.root / "raw"
        self.f_checkpoint = self.root / f"{self.model_name}_checkpoint.pt"
        self.f_plot = self.root / f"{self.model_name}_plot.svg"
        self.f_sgf = self.root / f"{self.model_name}_selfplay.sgf"

        # Train settings
        # --------------
        self.max_iters = max_iters
        self.log_interval = 1  # if <= 0, no log in the loop
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

        self.loss_fn = nn.CrossEntropyLoss()

        self.batch_size = 64
        self.lr = 1e-3

        # if True, restart training from checkpoint
        self.resume_from_checkpoint = True
        # if True, always save checkpoints after each eval
        self.always_save_checkpoint = True

        # Checkpoint data
        self.local_iter = 0
        self.eval_xs = []
        self.best_test_loss = float("+inf")
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []

        # Prepare dataset
        t0 = time.perf_counter()
        self.train_data = (
            MCTSDataset(
                self.root,
                train=True,
                download=True,  # disable download makes it faster
                transform=self.transform,
                target_transform=lambda y: self.transform(y).argmax(
                    1
                ),  # TODO: remove argmax
            )
            if train_data is None
            else train_data
        )

        self.test_data = (
            MCTSDataset(
                self.root,
                train=False,
                download=True,
                transform=self.transform,
                target_transform=lambda y: self.transform(y).argmax(1),
            )
            if test_data is None
            else test_data
        )
        t1 = time.perf_counter()
        dt = t1 - t0
        print()
        print(f"Load data time: {self.pretty_time(dt)}")

    def model_info(self, model):
        total_params = sum(p.numel() for p in model.parameters())

        print(f"Train on {self.device}")
        print(model)
        print(f"Parameters: {total_params:,d}")

    def get_batch(self, dataset):
        ix = torch.randint(len(dataset), (self.batch_size,))
        return dataset[ix]

    def eval_loss(self, dataset):
        loss, accuracy = 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.eval_iters):
                X, Y = self.get_batch(dataset)
                pred = self.model(X)
                lossi = self.loss_fn(pred, Y)
                acc = (pred.argmax(1) == Y).type(torch.float).mean().item()

                loss += lossi.item()
                accuracy += acc

        loss /= self.eval_iters
        accuracy /= self.eval_iters

        return loss, accuracy

    def final_log(self):
        final_train_loss, final_train_acc = self.eval_loss(self.train_data)
        final_test_loss, final_test_acc = self.eval_loss(self.test_data)

        print(
            "| Condition | Train Loss | Train Accuracy (%) | Test Loss | Test Accuracy (%) |"  # noqa: E501
        )
        print(
            "|----------:|-----------:|-------------------:|----------:|------------------:|"  # noqa: E501
        )
        print(
            f"| {self.max_iters:,d} iters | {final_train_loss:.3f} | {final_train_acc*100:.1f} | {final_test_loss:.3f} | {final_test_acc*100:.1f} |"  # noqa: E501
        )

    def plot_eval(self):
        xs = self.eval_xs
        l1, a1 = torch.tensor(self.train_losses), torch.tensor(self.train_accs)
        l2, a2 = torch.tensor(self.test_losses), torch.tensor(self.test_accs)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f'Training of "{self.model_name}"')

        axs[0].set_xlabel("Iter")
        axs[0].set_ylabel("Loss")
        axs[1].set_xlabel("Iter")
        axs[1].set_ylabel("Accuracy (%)")

        axs[0].plot(xs, l1, label="train")
        axs[0].plot(xs, l2, label="test")
        axs[0].legend()
        axs[1].plot(xs, a1 * 100, label="train")
        axs[1].plot(xs, a2 * 100, label="test")
        axs[1].legend()

        fig.savefig(self.f_plot)
        print(f"Save training plot at: {self.f_plot}")

    def save_checkpoint(self, i):
        torch.save(
            [
                self.model,
                i,
                self.eval_xs,
                self.best_test_loss,
                self.train_losses,
                self.train_accs,
                self.test_losses,
                self.test_accs,
            ],
            self.f_checkpoint,
        )

    def train(self):
        # Resume if needed
        if self.resume_from_checkpoint and self.f_checkpoint.is_file():
            [
                self.model,
                self.local_iter,
                self.eval_xs,
                self.best_test_loss,
                self.train_losses,
                self.train_accs,
                self.test_losses,
                self.test_accs,
            ] = torch.load(self.f_checkpoint, weights_only=False)
            print()
            print(f"Load iter {self.local_iter} checkpoint")

        # Train
        # -----
        torch.manual_seed(42)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        for i in range(self.local_iter, self.max_iters):
            t0 = time.perf_counter()

            self.model.train()
            self.optimizer.zero_grad()

            X, Y = self.get_batch(self.train_data)
            pred = self.model(X)
            loss = self.loss_fn(pred, Y)
            accuracy = (pred.argmax(1) == Y).type(torch.float).mean().item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            t1 = time.perf_counter()
            dt = t1 - t0

            if self.log_interval > 0 and i % self.log_interval == 0:
                print(
                    f"[{i/self.max_iters:6.1%}] {i:<7d}| loss {loss.item():<7.3f}| acc {accuracy:<7.1%}| time {self.pretty_time(dt)}"  # noqa: E501
                )

            if i % self.eval_interval == 0:
                t0 = time.perf_counter()

                self.model.eval()
                train_loss, train_acc = self.eval_loss(self.train_data)
                test_loss, test_acc = self.eval_loss(self.test_data)

                self.eval_xs.append(i)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.test_losses.append(test_loss)
                self.test_accs.append(test_acc)

                t1 = time.perf_counter()
                dt = t1 - t0
                print(
                    f"[{i/self.max_iters:6.1%}] {i:<7d}| loss {train_loss:<7.3f}| acc {train_acc:<7.1%}| test_loss {test_loss:<7.3f}| test_acc {test_acc:<7.1%}| time {self.pretty_time(dt)}"  # noqa: E501
                )

                if test_loss < self.best_test_loss:
                    self.best_test_loss = test_loss
                    self.save_checkpoint(i + 1)
                elif self.always_save_checkpoint:
                    self.save_checkpoint(i + 1)

        # Log
        print()
        self.final_log()
        print()
        self.plot_eval()

        # Sample
        # ------
        game = Game.new(self.board_size, komi=self.komi)
        agent = MLBot(self.model, self.encoder)

        n_moves = 0
        while not game.is_over:
            move = agent.select_move(game)
            game.apply_move(move)
            n_moves += 1

        sgf = game.to_pysgf().sgf()
        with open(self.f_sgf, "w", encoding="utf-8") as f:
            f.write(sgf)

        print()
        print(f"Generate {n_moves} moves")
        print(f"Save game at: {self.f_sgf}")


# TODO: add CLI
def main():
    trainer = ModelTrainer()
    return trainer.train()


if __name__ == "__main__":
    sys.exit(main())
