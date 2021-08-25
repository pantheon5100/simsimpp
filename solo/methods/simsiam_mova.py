import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseModel
from solo.losses.vicreg import covariance_loss

class SimSiamMova(BaseModel):
    def __init__(
        self,
        output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        normz:bool,
        random_z:bool,
        **kwargs,
    ):
        """Implements SimSiam (https://arxiv.org/abs/2011.10566).

        Args:
            output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)
        self.normz = normz
        self.random_z = random_z
        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
            # nn.BatchNorm1d(output_dim, affine=False),
        )
        # self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN


        # # predictor
        self.predictor = nn.Sequential()

        self.register_buffer("EXP_T", F.normalize(torch.randn(50000, output_dim), dim=-1))
        self.m = 0.8

        self.register_buffer("previouscentering", torch.randn(1, output_dim))
        self.register_buffer("onestepbeforecentering", torch.randn(1, output_dim))


    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimSiamMova, SimSiamMova).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simsiammova")

        # projector
        parser.add_argument("--output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--normz", action="store_true")

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        # loss
        parser.add_argument("--random_z", action="store_true")

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

        # prepare corresponding before
        img_indexes = batch[0]
        z_before = self.EXP_T[img_indexes, :].clone()

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        if self.normz:
            # print("normz")
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # ------- mova loss -------
        # neg_cos_sim = simsiam_loss_func(p1, z2) / 2 + simsiam_loss_func(p2, z1) / 2
        
        neg_cos_sim = F.mse_loss(z1, z_before)
        
        if self.random_z:
            # print("random")
            randn_z2 = F.normalize(2*torch.rand(self.batch_size, 2048)-1, dim=1).cuda()
            randn_z2 = (randn_z2.T* (torch.linalg.norm(z2, dim=1, ord=2))).T.detach()

            self.EXP_T[img_indexes] = self.m * randn_z2 + (1-self.m)*z2.detach()

        else:
            self.EXP_T[img_indexes] = self.m*self.EXP_T[img_indexes] + (1-self.m) * z2.detach()

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        with torch.no_grad():
            # normalize the vector to make it comparable
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

            centervector = ((z1 + z2)/2).mean(dim=0)
            residualvector = z2 - centervector
            # import pdb; pdb.set_trace()

            ZvsC = F.cosine_similarity(z2, centervector.expand(z2.size(0), 2048), dim=-1).mean()
            ZvsR = F.cosine_similarity(z2, residualvector, dim=-1).mean()
            CvsR = F.cosine_similarity(centervector.expand(z2.size(0), 2048), residualvector, dim=-1).mean()


            ratio_RvsW = (torch.linalg.norm(residualvector, dim=1, ord=2) / torch.linalg.norm(z2, dim=1, ord=2)).mean()
            ratio_CvsW = (torch.linalg.norm(centervector.expand(z2.size(0), 2048), dim=1, ord=2) / torch.linalg.norm(z2, dim=1, ord=2)).mean()

            CS1vsCc = F.cosine_similarity(self.onestepbeforecentering, centervector.reshape(1, -1))
            CS1minusCcvsCc = F.cosine_similarity(centervector.reshape(1, -1)-self.onestepbeforecentering  , centervector.reshape(1, -1))
            CS1minusCcvsCS1 = F.cosine_similarity(centervector.reshape(1, -1)-self.onestepbeforecentering  , self.onestepbeforecentering)


            # self.recod_epoch[self.trainer.global_step - self.trainer.current_epoch * 195] = CS1minusCcvsCc.cpu()
            # CS1minusCcvsCc = F.cosine_similarity(self.onestepbeforecentering, centervector.reshape(1, -1))

            # if self.trainer.is_last_batch:
            #     import numpy as np
            #     np.savetxt( f"BS{self.trainer.current_epoch}.txt", self.recod_epoch.numpy(),)

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
            cov_loss = covariance_loss(z1, z2)
            mean_z = (z1.abs().mean(dim=1) + z2.abs().mean(dim=1)).mean()/2

            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            norm_cov_loss = covariance_loss(z1, z2)

            norm_mean_z = (z1.abs().mean(dim=1) + z2.abs().mean(dim=1)).mean()/2

        metrics = {
            "neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "cov_loss": cov_loss,
            "norm_cov_loss": norm_cov_loss,
            "mean_z": mean_z,
            "norm_mean_z": norm_mean_z,
        }

        metrics.update(new_metric_log)

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
