from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule

from src import utils as U


class LowCostRobotACTBCModule(LightningModule):
    def __init__(
        self,
        policy,
        optimizer,
        lr_scheduler,
        train_metrics,
        val_metrics,
        best_val_metrics,
        compile: bool = False,
        param_dicts: list = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["policy", "train_metrics", "val_metrics", "best_val_metrics"],
        )

        self.policy = policy

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        # for tracking best so far validation metrics
        self.best_val_metrics = best_val_metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.best_val_metrics.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy(batch)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss_dict = self.model_step(batch)

        # loss_dict['loss'].backward()
        # for name, param in self.policy.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # exit()

        # update and log metrics
        self.train_metrics(loss_dict)
        self.log_dict(
            self.train_metrics.metrics_dict(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["actions"].shape[0],
        )

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        return super().on_train_epoch_end()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss_dict = self.model_step(batch)

        # update and log metrics
        self.val_metrics(loss_dict)
        self.log_dict(
            self.val_metrics.metrics_dict(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["actions"].shape[0],
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metrics = self.val_metrics.compute()  # get current val metrics
        self.best_val_metrics(metrics)  # update best so far val metrics
        # log `best_val_metrics` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log_dict(self.best_val_metrics.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.policy = torch.compile(self.policy)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = U.build_optimizer(
            self.hparams.optimizer, self.policy, self.hparams.param_dicts
        )
        if self.hparams.lr_scheduler is not None:
            self.hparams.lr_scheduler.scheduler.total_steps = (
                self.trainer.estimated_stepping_batches
            )
            scheduler = U.build_scheduler(
                self.hparams.lr_scheduler.scheduler, optimizer=optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.lr_scheduler.get("monitor", "val/loss"),
                    "interval": self.hparams.lr_scheduler.get("interval", "step"),
                    "frequency": self.hparams.lr_scheduler.get("frequency", 1),
                },
            }
        return {"optimizer": optimizer}
