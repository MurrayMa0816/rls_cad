from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .environment import (
    JointMobilitySupportEnv,
    MapConfig,
    TheaterHexEnvConfig,
    TheaterHexResourceEnv,
    softmax_np,
    sparse_budget_projection,
    sparsemax_np,
)


@dataclass
class TMSBDActionConfig:
    tau_min: float = 0.08
    fixed_topk_ratio: float = 0.10
    min_effective_share: float = 1e-3
    action_box_low: float = -8.0
    action_box_high: float = 8.0


@dataclass
class ROIParamActionConfig:
    topk_ratio: float = 0.10
    action_box_low: float = -8.0
    action_box_high: float = 8.0


@dataclass
class ROISoftmaxHybridActionConfig:
    topk_ratio: float = 0.10
    action_box_low: float = -8.0
    action_box_high: float = 8.0


@dataclass
class DualExpertROIParamActionConfig:
    action_box_low: float = -8.0
    action_box_high: float = 8.0


@dataclass
class AdaptiveDecoderSelectionActionConfig:
    action_box_low: float = -8.0
    action_box_high: float = 8.0


@dataclass
class AdaptiveDecoderSelectionResidualActionConfig:
    action_box_low: float = -8.0
    action_box_high: float = 8.0


@dataclass
class MorphologyGuidedSoftmaxActionConfig:
    action_box_low: float = -8.0
    action_box_high: float = 8.0


def numpy_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax_np(logits, temperature=temperature)


class _ModeEnv(JointMobilitySupportEnv):
    allocation_mode: str = "tmsbd"

    def __init__(self, config: TheaterHexEnvConfig | MapConfig | None = None, **_: object):
        if config is None:
            cfg = TheaterHexEnvConfig()
        elif isinstance(config, TheaterHexEnvConfig):
            cfg = config
        elif isinstance(config, MapConfig):
            cfg = TheaterHexEnvConfig(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)!r}")
        cfg.map_config.allocation_mode = self.allocation_mode
        super().__init__(cfg)


class TheaterHexSoftmaxActionEnv(_ModeEnv):
    allocation_mode = "direct_softmax"


class TheaterHexSparseProjectionActionEnv(_ModeEnv):
    allocation_mode = "direct_sparse_projection"


class TheaterHexDirectSimplexActionEnv(_ModeEnv):
    allocation_mode = "direct_simplex"


class TheaterHexROIParamActionEnv(_ModeEnv):
    allocation_mode = "roi_param"


class TheaterHexROISoftmaxHybridActionEnv(_ModeEnv):
    allocation_mode = "tmsbd_softmax_budget"


class TheaterHexDualExpertROIParamActionEnv(_ModeEnv):
    allocation_mode = "coverage_focus_dual"


class TheaterHexAdaptiveDecoderSelectionEnv(_ModeEnv):
    allocation_mode = "tmsbd_no_gate"


class TheaterHexAdaptiveDecoderSelectionResidualEnv(_ModeEnv):
    allocation_mode = "tmsbd"


class TheaterHexMorphologyGuidedSoftmaxActionEnv(_ModeEnv):
    allocation_mode = "tmsbd"


class TheaterHexTMSBDEnv(_ModeEnv):
    allocation_mode = "tmsbd"


class TheaterHexTMSBDNoGateEnv(_ModeEnv):
    allocation_mode = "tmsbd_no_gate"


class TheaterHexTMSBDNoChainEnv(_ModeEnv):
    allocation_mode = "tmsbd_no_chain"


class TheaterHexTMSBDSoftmaxGateEnv(_ModeEnv):
    allocation_mode = "tmsbd_softmax_gate"


class TheaterHexTMSBDSoftmaxBudgetEnv(_ModeEnv):
    allocation_mode = "tmsbd_softmax_budget"


class TheaterHexTMSBDFixedTopKEnv(_ModeEnv):
    allocation_mode = "tmsbd_fixed_topk"


class TheaterHexTMSBDFixedMorphologyEnv(_ModeEnv):
    allocation_mode = "tmsbd_fixed_morphology"


class TheaterHexTMSBDSingleCriticalEnv(_ModeEnv):
    allocation_mode = "tmsbd_single_critical"


class TheaterHexTMSBDSingleSupportEnv(_ModeEnv):
    allocation_mode = "tmsbd_single_support"


class TheaterHexTMSBDSingleBacklogEnv(_ModeEnv):
    allocation_mode = "tmsbd_single_backlog"


class TheaterHexTMSBDSingleE2EEnv(_ModeEnv):
    allocation_mode = "tmsbd_single_e2e"


class TheaterHexLTSSCPLatentDecoderEnv(_ModeEnv):
    allocation_mode = "lts_scp_latent"


class TheaterHexSignalMatchedSoftmaxEnv(_ModeEnv):
    allocation_mode = "signal_matched_softmax"


class TheaterHexSignalMatchedSparseProjectionEnv(_ModeEnv):
    allocation_mode = "signal_matched_sparse_projection"


__all__ = [
    "TMSBDActionConfig",
    "ROIParamActionConfig",
    "ROISoftmaxHybridActionConfig",
    "DualExpertROIParamActionConfig",
    "AdaptiveDecoderSelectionActionConfig",
    "AdaptiveDecoderSelectionResidualActionConfig",
    "MorphologyGuidedSoftmaxActionConfig",
    "TheaterHexResourceEnv",
    "TheaterHexSoftmaxActionEnv",
    "TheaterHexSparseProjectionActionEnv",
    "TheaterHexDirectSimplexActionEnv",
    "TheaterHexROIParamActionEnv",
    "TheaterHexROISoftmaxHybridActionEnv",
    "TheaterHexDualExpertROIParamActionEnv",
    "TheaterHexAdaptiveDecoderSelectionEnv",
    "TheaterHexAdaptiveDecoderSelectionResidualEnv",
    "TheaterHexMorphologyGuidedSoftmaxActionEnv",
    "TheaterHexTMSBDEnv",
    "TheaterHexTMSBDNoGateEnv",
    "TheaterHexTMSBDNoChainEnv",
    "TheaterHexTMSBDSoftmaxGateEnv",
    "TheaterHexTMSBDSoftmaxBudgetEnv",
    "TheaterHexTMSBDFixedTopKEnv",
    "TheaterHexTMSBDFixedMorphologyEnv",
    "TheaterHexTMSBDSingleCriticalEnv",
    "TheaterHexTMSBDSingleSupportEnv",
    "TheaterHexTMSBDSingleBacklogEnv",
    "TheaterHexTMSBDSingleE2EEnv",
    "TheaterHexLTSSCPLatentDecoderEnv",
    "TheaterHexSignalMatchedSoftmaxEnv",
    "TheaterHexSignalMatchedSparseProjectionEnv",
    "numpy_softmax",
    "sparse_budget_projection",
    "sparsemax_np",
]
