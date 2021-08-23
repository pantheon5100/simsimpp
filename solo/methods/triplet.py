import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseModel
from solo.losses.vicreg import covariance_loss

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()

    def forward(self,x):

        return x

class SimSiam_Triplet(BaseModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        BL:bool,
        experiment:str,
        **kwargs,
    ):
        """Implements SimSiam (https://arxiv.org/abs/2011.10566).

        Args:
            output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)

        self.experiment = experiment

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # predictor
        if not BL:
            self.predictor = nn.Sequential(
                nn.Linear(output_dim, pred_hidden_dim, bias=False),
                nn.BatchNorm1d(pred_hidden_dim),
                nn.ReLU(),
                nn.Linear(pred_hidden_dim, output_dim),
            )
        elif BL:
            self.predictor = nn.Sequential(BiasLayer())
        
        self.register_buffer("previouscentering", torch.randn(1, output_dim))
        self.register_buffer("onestepbeforecentering", torch.randn(1, output_dim))

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimSiam_Triplet, SimSiam_Triplet).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("SimSiam_Triplet")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--BL", action="store_true")

        SUPPORTED_EXP = ["Erc", "E1c", "E05c", "E03c", "E01c", "E1r", "E05r", "Regc", "Regr"]
        parser.add_argument("--experiment", choices=SUPPORTED_EXP, type=str)

        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        # training with triplet loss
        parser.add_argument("--triplet", action="store_true")
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params: List[dict] = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters(), "static_lr": True},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the encoder, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimSiam reusing BaseModel training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images
            batch_idx (int): index of the batch

        Returns:
            torch.Tensor: total loss composed of SimSiam loss and classification loss
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2, feats3 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)
        z3 = self.projector(feats3)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        p3 = self.predictor(z3)

        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)
        z3_norm = F.normalize(z3, dim=-1)

        batch_zie = int(z1.size(0)/2)

        # ------- Triplet loss -------
        # ["Erc", "E1c", "E05c", "E03c", "E01c", "E1r", "E05r", "Regc", "Regr"]
        def loss_function(p, z, c):
            if self.experiment == "Erc":
                # print("ecr")
                return -(p*z).sum(dim=1).mean() + (p*c).sum(dim=1).mean()
            elif self.experiment == "E1c":
                # print("E1c")
                return -(p*z).sum(dim=1).mean() + (p*(c.mean(dim=0))).sum(dim=1).mean()
            elif self.experiment == "E05c":
                return -(p*z).sum(dim=1).mean() + (p*(0.5*c.mean(dim=0))).sum(dim=1).mean()
            elif self.experiment == "E03c":
                return -(p*z).sum(dim=1).mean() + (p*(0.3*c.mean(dim=0))).sum(dim=1).mean()
            elif self.experiment == "E01c":
                return -(p*z).sum(dim=1).mean() + (p*(0.1*c.mean(dim=0))).sum(dim=1).mean()
            elif self.experiment == "E1r":
                return -(p*z).sum(dim=1).mean() + (p*(c-c.mean(dim=0))).sum(dim=1).mean()
            elif self.experiment == "E05r":
                return -(p*z).sum(dim=1).mean() + (p*((c-c.mean(dim=0))/2)).sum(dim=1).mean()
            elif self.experiment == "Regc":
                return -(p*z).sum(dim=1).mean() + (torch.linalg.norm(c.mean(dim=0), dim=-1, ord=2)).sum(dim=-1).mean()
            elif self.experiment == "Regr":
                return -(p*z).sum(dim=1).mean() + (torch.linalg.norm(c-c.mean(dim=0), dim=-1, ord=2)).sum(dim=-1).mean()

        neg_cos_sim = loss_function(z1_norm[:batch_zie], z2_norm[:batch_zie], z3_norm[batch_zie:]) / 2 + loss_function(z1_norm[batch_zie:], z2_norm[batch_zie:], z3_norm[:batch_zie]) / 2

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        with torch.no_grad():
            # normalize the vector to make it comparable
            z2 = F.normalize(z2, dim=-1)
            z1 = F.normalize(z1, dim=-1)

            centervector = ((z1_norm + z2_norm + z3_norm)/2).mean(dim=0)
            residualvector = z2_norm - centervector
            # import pdb; pdb.set_trace()

            ZvsC = F.cosine_similarity(z2_norm, centervector.expand(z2_norm.size(0), 2048), dim=-1).mean()
            ZvsR = F.cosine_similarity(z2_norm, residualvector, dim=-1).mean()
            CvsR = F.cosine_similarity(centervector.expand(z2_norm.size(0), 2048), residualvector, dim=-1).mean()


            ratio_RvsW = (torch.linalg.norm(residualvector, dim=1, ord=2) / torch.linalg.norm(z2_norm, dim=1, ord=2)).mean()
            ratio_CvsW = (torch.linalg.norm(centervector.expand(z2_norm.size(0), 2048), dim=1, ord=2) / torch.linalg.norm(z2_norm, dim=1, ord=2)).mean()

            CS1vsCc = F.cosine_similarity(self.onestepbeforecentering, centervector.reshape(1, -1))
            CS1minusCcvsCc = F.cosine_similarity(centervector.reshape(1, -1)-self.onestepbeforecentering  , centervector.reshape(1, -1))
            CS1minusCcvsCS1 = F.cosine_similarity(centervector.reshape(1, -1)-self.onestepbeforecentering  , self.onestepbeforecentering)

            self.onestepbeforecentering = centervector.reshape(1, -1)

            new_metric_log={"ZvsC_norm":ZvsC,
                    "ZvsR_norm":ZvsR,
                    "ratio_RvsW_norm":ratio_RvsW,
                    "ZvsR_norm":ZvsR,
                    "ratio_CvsW_norm":ratio_CvsW,
                    "CvsR_norm":CvsR,
                    "CS1vsCc":CS1vsCc,
                    "CS1minusCcvsCc":CS1minusCcvsCc,
                    "CS1minusCcvsCS1":CS1minusCcvsCS1,
                    }

            if self.trainer.global_step % 100 == 0:

                CpvsCc = F.cosine_similarity(self.previouscentering, centervector.reshape(1, -1))

                self.previouscentering = centervector.reshape(1, -1).clone()

                new_metric_log.update({"CpvsCc_norm": CpvsCc})

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        with torch.no_grad():
            cov_loss = covariance_loss(z1_norm, z2_norm)
            mean_z = (z1_norm.abs().mean(dim=1) + z2_norm.abs().mean(dim=1)).mean()/2

        metrics = {
            "neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "cov_loss": cov_loss,
            "mean_z": mean_z,
        }

        metrics.update(new_metric_log)

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
