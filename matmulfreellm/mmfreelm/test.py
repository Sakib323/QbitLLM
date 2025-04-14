# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from matmulfreellm.mmfreelm.layers.hgrn_bit import HGRNBitAttention
from matmulfreellm.mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from matmulfreellm.mmfreelm.models.utils import RecurrentCache
from matmulfreellm.mmfreelm.modules import FusedCrossEntropyLoss, RMSNorm  
from mmfreelm.modules.activations import swiglu_linear, swiglu
#from mmfreelm.ops.bitnet import BitLinear_Fuse as BitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear
 


print("Everything is ok")