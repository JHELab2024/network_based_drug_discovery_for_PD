"""
VecNet as an nn.Module, refactored so the same object serves inference and
fine-tuning.

    raw_target(100) --/-mean_t /len_t-> target_proj(100->2048) -relu-\
                                                                       cat(4096) -> head(4096->512->512->1) -> logit
    raw_drug(300)  --/-mean_d /len_d-> drug_proj (300->2048) -relu-/

- The four normalization constants are frozen buffers (register_buffer). They
  define the coordinate system the pretrained weights live in and are NEVER
  recomputed. Normalization is done in float64 (matching the original featurizer),
  then cast to float32 for the linear stack -> bit-identical to the baseline.
- forward() returns a LOGIT (no sigmoid) so training can use BCEWithLogitsLoss;
  callers apply sigmoid for probabilities.

Fine-tune rung (set_finetune_mode): target_proj frozen, drug_proj + head trainable,
buffers never trainable.
"""
import numpy as np
import torch
import torch.nn as nn

# map npz bundle keys -> module parameters
_LINEARS = {"target_proj": "fc_target", "drug_proj": "fc_drug"}
_HEAD = {0: "fc1", 2: "fc2", 4: "fc_out"}   # indices into the head Sequential


class VecNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_proj = nn.Linear(100, 2048)                 # frozen at fine-tune
        self.drug_proj = nn.Linear(300, 2048)                   # trainable
        self.head = nn.Sequential(                              # trainable
            nn.Linear(4096, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )
        # frozen normalization constants (float64), never recomputed
        self.register_buffer("mean_drug", torch.zeros(300, dtype=torch.float64))
        self.register_buffer("len_drug", torch.ones((), dtype=torch.float64))
        self.register_buffer("mean_target", torch.zeros(100, dtype=torch.float64))
        self.register_buffer("len_target", torch.ones((), dtype=torch.float64))

    def forward(self, raw_target, raw_drug):
        # normalize in float64 with the frozen buffers, then float32 through the stack
        nt = ((raw_target.double() - self.mean_target) / self.len_target).float()
        nd = ((raw_drug.double() - self.mean_drug) / self.len_drug).float()
        ht = torch.relu(self.target_proj(nt))
        hd = torch.relu(self.drug_proj(nd))
        return self.head(torch.cat([ht, hd], dim=-1))           # logit, shape (N,1)

    # ---- weight <-> npz bundle (pretrained format) ----
    def load_bundle(self, npz, run):
        def t(a):
            return torch.tensor(np.asarray(a), dtype=torch.float32)
        for attr, key in _LINEARS.items():
            lin = getattr(self, attr)
            lin.weight.data.copy_(t(npz["run%d.%s.weight" % (run, key)]))
            lin.bias.data.copy_(t(npz["run%d.%s.bias" % (run, key)]))
        for idx, key in _HEAD.items():
            self.head[idx].weight.data.copy_(t(npz["run%d.%s.weight" % (run, key)]))
            self.head[idx].bias.data.copy_(t(npz["run%d.%s.bias" % (run, key)]))
        self.mean_drug.copy_(torch.tensor(np.asarray(npz["const.mean_drug"]), dtype=torch.float64))
        self.len_drug.copy_(torch.tensor(float(npz["const.len_drug"]), dtype=torch.float64))
        self.mean_target.copy_(torch.tensor(np.asarray(npz["const.mean_target"]), dtype=torch.float64))
        self.len_target.copy_(torch.tensor(float(npz["const.len_target"]), dtype=torch.float64))
        return self

    def fold_state(self, run):
        """This fold's parameters + constants as npz keys (pretrained bundle format)."""
        out = {}
        for attr, key in _LINEARS.items():
            lin = getattr(self, attr)
            out["run%d.%s.weight" % (run, key)] = lin.weight.detach().cpu().numpy().astype(np.float32)
            out["run%d.%s.bias" % (run, key)] = lin.bias.detach().cpu().numpy().astype(np.float32)
        for idx, key in _HEAD.items():
            out["run%d.%s.weight" % (run, key)] = self.head[idx].weight.detach().cpu().numpy().astype(np.float32)
            out["run%d.%s.bias" % (run, key)] = self.head[idx].bias.detach().cpu().numpy().astype(np.float32)
        return out

    def constants_state(self):
        return {
            "const.mean_drug": self.mean_drug.detach().cpu().numpy().astype(np.float64),
            "const.len_drug": np.float64(self.len_drug.item()),
            "const.mean_target": self.mean_target.detach().cpu().numpy().astype(np.float64),
            "const.len_target": np.float64(self.len_target.item()),
        }

    # ---- fine-tune rung 3 ----
    def set_finetune_mode(self):
        self.target_proj.requires_grad_(False)   # keep target side model-general
        self.drug_proj.requires_grad_(True)      # re-project the user's chemistry
        self.head.requires_grad_(True)           # move the decision boundary
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
