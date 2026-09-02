# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The concrete Engine implementation using PyTorch TorchTitan parallelism (FSDP2 + TP + PP)
"""

import gc
import importlib
import logging
import os
import re
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.context_parallel import prepare_context_parallel_input

from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.train import Trainer

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.megatron_peft_utils import add_base_layer_suffix
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig
from verl.workers.engine.torchtitan.utils import (
    NoOpDataLoader,
    derive_torchtitan_name_and_flavor,
    enable_fsdp_gradient_division,
    get_attention_masks,
    iter_per_tensor_params_ep,
)

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class _PipelineLossBridge:
    """The loss the pipeline schedule calls on the last stage.

    torchtitan builds its schedule with a loss at construction; verl hands a
    loss function to every forward_backward_batch call. The bridge is
    installed as the schedule's loss once and re-targeted per call: the
    target the schedule passes is the micro-batch index, and the bridge runs
    verl's output preparation and loss on that micro-batch, keeping the
    outputs the worker collects from the last stage.
    """

    def __init__(self, engine):
        self.engine = engine
        self.reset(None)

    def reset(self, loss_function):
        self.loss_function = loss_function
        self.micro_batches = {}
        self.output_args = {}
        self.outputs = {}

    def __call__(self, pred, target, **_):
        index = int(target.item()) if torch.is_tensor(target) else int(target)
        engine = self.engine
        pred = engine._finish_pred(pred)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        micro_batch = self.micro_batches[index]
        model_output = engine.prepare_model_outputs(
            logits=pred, output_args=self.output_args[index], micro_batch=micro_batch
        )
        if self.loss_function is not None:
            loss, metrics = self.loss_function(
                model_output=model_output, data=micro_batch, dp_group=engine.get_data_parallel_group()
            )
        else:
            loss = pred.new_zeros((), dtype=torch.float32)
            metrics = {}
        self.outputs[index] = {"model_output": model_output, "loss": loss.detach().item(), "metrics": metrics}
        return loss


class TorchTitanEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch TorchTitan parallelism.

    Supports model sharding with FSDP2, tensor parallelism, activation/optimizer offloading,
    LoRA, and sequence parallelism following the TorchTitan design.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: TorchtitanEngineConfig,
        optimizer_config: TorchtitanOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the TorchTitanEngine.

        Sets up distributed device meshes for tensor and data parallelism, LoRA, and offload policies.

        Args:
            model_config: Configuration for HuggingFace model.
            engine_config: Configuration for FSDP/TorchTitan engine (uses FSDP2).
            optimizer_config: Configuration for optimizer.
            checkpoint_config: Configuration for checkpointing.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        # Derive torchtitan model name and flavor from HF config
        torchtitan_name, torchtitan_flavor = derive_torchtitan_name_and_flavor(self.model_config.hf_config)
        # kimi_k3 handles CP module-internally (Ulysses) and is causal-only:
        # no attention_masks consumed, no upstream CP mask sharding needed.
        self._model_cp_is_module_internal = torchtitan_name == "kimi_k3"

        # Get ModelSpec from model registry
        from .utils import _import_torchtitan_model_module

        model_module = _import_torchtitan_model_module(torchtitan_name)
        # Two naming spaces in the kimi_k3 package: model_registry parses
        # "<size>_<variant>", while the debug, report-architecture and QAT
        # flavors are config_registry FUNCTIONS whose names it cannot parse.
        # Without the fallback, VERL_TORCHTITAN_FLAVOR can only reach the first
        # kind, which silently excludes every flavor defined the second way.
        try:
            model_spec = model_module.model_registry(
                torchtitan_flavor, attn_backend=self.engine_config.attn_type
            )
        except (ValueError, KeyError):
            # model_registry raises KeyError for a name its parser does not
            # know and ValueError for one it parses but does not have; the
            # config_registry functions are reached through either.
            import importlib

            try:
                config_registry = importlib.import_module(
                    f"{model_module.__name__}.config_registry"
                )
            except ImportError:
                config_registry = None
            fn = getattr(config_registry, torchtitan_flavor, None) if config_registry else None
            if fn is None or not callable(fn):
                raise
            model_spec = fn().model_spec

        optimizer = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name=self.optimizer_config.name,
                    optimizer_kwargs={
                        "lr": self.optimizer_config.lr,
                        "eps": self.optimizer_config.eps,
                        "betas": (self.optimizer_config.betas[0], self.optimizer_config.betas[1]),
                        "weight_decay": self.optimizer_config.weight_decay,
                    },
                )
            ],
        )

        total_steps = self.optimizer_config.total_training_steps
        lr_warmup_steps = self.optimizer_config.lr_warmup_steps
        if lr_warmup_steps is None or lr_warmup_steps <= 0:
            lr_warmup_steps = int(self.optimizer_config.lr_warmup_steps_ratio * total_steps)

        lr_scheduler = LRSchedulersContainer.Config(
            warmup_steps=lr_warmup_steps,
            decay_type=self.optimizer_config.decay_type,
            min_lr_factor=self.optimizer_config.min_lr_factor,
        )
        parallelism = ParallelismConfig(
            data_parallel_replicate_degree=self.engine_config.data_parallel_replicate_size,
            data_parallel_shard_degree=self.engine_config.data_parallel_shard_size,
            fsdp_reshard_after_forward=self.engine_config.reshard_after_forward,
            tensor_parallel_degree=self.engine_config.tensor_parallel_size,
            pipeline_parallel_degree=self.engine_config.pipeline_parallel_size,
            context_parallel_degree=self.engine_config.context_parallel_size,
            expert_parallel_degree=self.engine_config.expert_parallel_size,
            spmd_backend=self.engine_config.spmd_backend,
            # kimi_k3's module-internal CP reassembles contiguous
            # rank-ordered seq shards; the upstream default 'headtail'
            # balancer PERMUTES the sequence before sharding, silently
            # breaking causal order (future-token leakage) -- its
            # parallelize raises on any balancer. Upstream-CP models
            # (llama3/qwen3/...) keep the torchtitan default.
            context_parallel_load_balancer=(
                None if torchtitan_name == "kimi_k3"
                else ParallelismConfig.context_parallel_load_balancer
            ),
        )
        checkpoint = CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            initial_load_path=model_config.path,
            # verl's trainer.save_freq is the cadence authority and save() is
            # only called on those steps; defer torchtitan's own interval to 1
            # so every requested save writes (default 500 silently drops all
            # saves in runs shorter than 500 / not multiples of it).
            interval=1,
        )
        compile_config = CompileConfig(enable=self.engine_config.use_torch_compile)
        training_kwargs = {}
        # The kimi_k3 standard token dispatcher synchronizes with the CPU,
        # which CUDA graphs cannot capture; run eager under expert parallel.
        training_kwargs["disable_cuda_graphs"] = True
        if self.engine_config.max_seq_len is not None:
            training_kwargs["seq_len"] = self.engine_config.max_seq_len
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            training = TrainingConfig(enable_cpu_offload=True, **training_kwargs)
        else:
            training = TrainingConfig(**training_kwargs)

        # Activation checkpointing mode. Note: under spmd_backend="spmd_types" with
        # eager execution (use_torch_compile=False), selective/full AC recompute runs
        # on the autograd backward thread where the thread-local SPMD mesh is inactive,
        # so spmd.assert_type() raises "no current mesh". Set activation_checkpoint="none"
        # (or enable torch.compile, which recomputes in-graph) in that configuration.
        activation_checkpoint = {
            "selective": SelectiveAC.Config,
            "full": FullAC.Config,
            "none": lambda: None,
        }[self.engine_config.activation_checkpoint]()

        # Construct Torchtitan's Trainer.Config
        self.config = Trainer.Config(
            model_spec=model_spec,
            hf_assets_path=self.model_config.path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            parallelism=parallelism,
            checkpoint=checkpoint,
            compile=compile_config,
            training=training,
            activation_checkpoint=activation_checkpoint,
            # Use a no-op dataloader since verl has its own data loading
            dataloader=NoOpDataLoader.Config(),
            # Provide a concrete loss so Trainer.__init__ can build it;
            # verl uses its own loss function and ignores this one.
            loss=CrossEntropyLoss.Config(),
        )
        self.trainer = Trainer(self.config)

        self._init_device_mesh()

        # Re-enable FSDP's gradient division for verl's loss scaling.
        # TorchTitan disables gradient division by default (for global token normalization),
        # but verl's loss function multiplies by dp_size to compensate for gradient averaging.
        if self.engine_config.data_parallel_shard_size > 1:
            dp_size = self.get_data_parallel_size()
            for model_part in self.trainer.model_parts:
                enable_fsdp_gradient_division(model_part, dp_size)

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        is_collect = True
        # TP: outputs are on TP rank 0
        if self.parallel_dims.tp > 1:
            tp_mesh = self.parallel_dims.get_optional_mesh("tp")
            is_collect = is_collect and (tp_mesh.get_local_rank() == 0)
        # PP: outputs are on the last PP rank
        if self.parallel_dims.pp > 1:
            pp_mesh = self.parallel_dims.get_optional_mesh("pp")
            is_collect = is_collect and (pp_mesh.get_local_rank() == self.parallel_dims.pp - 1)
        # CP: outputs are on CP rank 0
        if self.parallel_dims.cp > 1:
            cp_mesh = self.parallel_dims.get_optional_mesh("cp")
            is_collect = is_collect and (cp_mesh.get_local_rank() == 0)
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler with TorchTitan parallelism.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager.
        """
        self.module = self.trainer.model_parts
        if self.parallel_dims.pp_enabled:
            # torchtitan's schedule owns the loss; hand it the bridge so verl's
            # per-call loss function reaches the last stage.
            self._pp_bridge = _PipelineLossBridge(self)
            self.trainer.pp_schedule._loss_fn = self._pp_bridge
        self.checkpointer = self.trainer.checkpointer
        # load initial HF weights
        self.checkpointer.load()

        if not self.engine_config.forward_only:
            self.optimizer = self.trainer.optimizers
            self.lr_scheduler = self.trainer.lr_schedulers
        else:
            self.optimizer = None
            self.lr_scheduler = None

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _init_device_mesh(self):
        """Initialize the device mesh for TorchTitan style parallelism."""
        world_size = torch.distributed.get_world_size()
        self.parallel_dims = ParallelDims(
            dp_shard=self.engine_config.data_parallel_shard_size,
            dp_replicate=self.engine_config.data_parallel_replicate_size,
            cp=self.engine_config.context_parallel_size,
            tp=self.engine_config.tensor_parallel_size,
            pp=self.engine_config.pipeline_parallel_size,
            ep=self.engine_config.expert_parallel_size,
            world_size=world_size,
        )
        self.device_mesh = self.parallel_dims.build_mesh()

        # Mirror torchtitan's init_distributed (which verl bypasses): disable autograd
        # multithreading so backward-thread activation-checkpoint recompute can access the
        # thread-local SPMD mesh / process groups (e.g. current_spmd_mesh().get_group(...)).
        torch.autograd.set_multithreading_enabled(False)

    def train_mode(self, **kwargs):
        """Return a context manager for training mode."""
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """Return a context manager for evaluation mode."""
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        mesh = self._get_data_parallel_mesh()
        return 0 if mesh is None else mesh.get_local_rank()

    def get_data_parallel_size(self):
        return self.engine_config.data_parallel_shard_size * self.engine_config.data_parallel_replicate_size

    def get_data_parallel_group(self):
        mesh = self._get_data_parallel_mesh()
        if mesh is not None:
            return mesh.get_group()
        # If world_size == dp_size (e.g. single GPU, or all ranks are DP),
        # return WORLD so that collective ops in _postprocess_output
        # (allgather_dict_into_dict, all_reduce) still run and produce the
        # correct metric aggregation format.
        if torch.distributed.get_world_size() == self.get_data_parallel_size():
            return torch.distributed.group.WORLD
        return None

    def get_model_parallel_group(self):
        raise NotImplementedError

    def get_context_parallel_group(self):
        raise NotImplementedError

    def _get_data_parallel_mesh(self):
        """Get the DATA-loading parallel mesh (excludes cp).

        torchtitan's "fsdp" mesh is dp_shard x cp -- FSDP folds cp into its
        gradient-reduction axis -- so under cp > 1 it is NOT the dataloader
        axis: all cp ranks of one dp group must receive the SAME batch (the
        trainer seq-shards it across cp afterwards). "batch" is
        dp_replicate x dp_shard, exactly the sampler axis; using "fsdp"
        here made rank cp_i draw sample shard i and crash the
        DistributedSampler (rank >= num_replicas) at dp_shard=1, cp=2.
        """
        mesh = self.parallel_dims.get_optional_mesh("batch")
        if mesh is None and self.parallel_dims.cp == 1:
            # Legacy fallbacks; only valid when cp == 1 (then fsdp ==
            # dp_shard). Under cp > 1 with no real dp, the correct answer
            # is None (single dp replica, rank 0).
            mesh = self.parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"])
            if mesh is None:
                mesh = self.parallel_dims.get_optional_mesh("fsdp")
            if mesh is None:
                mesh = self.parallel_dims.get_optional_mesh("dp_replicate")
        return mesh

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """Perform forward and optionally backward pass on a batch."""
        tu.assign_non_tensor(data, sp_size=self.engine_config.tensor_parallel_size)

        # Compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        dp_group = self.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )

        if self.parallel_dims.pp_enabled:
            output_lst = self._pp_forward_backward_batch(
                micro_batches, loss_function=loss_function, forward_only=forward_only
            )
            return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)
        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        # train_context activates the (thread-local) SPMD mesh required by spmd_types; it must
        # span backward too, since activation-checkpoint recompute re-runs the forward there.
        for micro_batch in micro_batches:
            with self.trainer.train_context(), ctx:
                loss, output = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    loss.backward()
            output_lst.append(output)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def _pp_forward_backward_batch(self, micro_batches, *, loss_function, forward_only):
        """Run the pipeline schedule once per verl micro-batch.

        The schedule is built with one pipeline microbatch, so every verl
        micro-batch is one schedule step: no fill/drain overlap, but the
        numbers are the same for any micro-batch count, including dynamic
        batch sizes. The last stage's outputs come back through the bridge;
        the other stages return placeholders the worker never collects.
        """
        trainer = self.trainer
        schedule = trainer.pp_schedule
        if schedule._n_microbatches != 1:
            raise ValueError(
                "the verl torchtitan engine drives the pipeline one micro-batch per step; "
                f"parallelism.num_pp_microbatches must be 1, got {schedule._n_microbatches}"
            )
        bridge = self._pp_bridge
        bridge.reset(loss_function)
        device_name = get_device_name()
        first, last = trainer.pp_has_first_stage, trainer.pp_has_last_stage
        folded = getattr(self.module[0], "folded_token_stream", False)
        output_lst = []
        for index, micro_batch in enumerate(micro_batches):
            micro_batch = micro_batch.to(get_device_id())
            input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)
            if folded and input_ids.dim() == 2 and input_ids.shape[0] == 1:
                input_ids = input_ids.squeeze(0)
                extra_inputs = {
                    k: (v.squeeze(0) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 else v)
                    for k, v in extra_inputs.items()
                }
            bridge.micro_batches[index] = micro_batch
            bridge.output_args[index] = output_args
            kwargs = {**extra_inputs, **extra_kwargs}
            target = torch.tensor(index, device=get_device_id())
            losses = [] if last else None
            with trainer.train_context(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                if forward_only:
                    schedule.eval(
                        arg_mbs=[(input_ids,)] if first else None,
                        kwarg_mbs=[kwargs],
                        target_mbs=[target] if last else None,
                        losses=losses,
                    )
                else:
                    schedule.step(
                        arg_mbs=[(input_ids,)] if first else None,
                        kwarg_mbs=[kwargs],
                        target_mbs=[target] if last else None,
                        losses=losses,
                        return_outputs=False,
                    )
            output_lst.append(bridge.outputs.get(index, {"loss": 0.0, "metrics": {}}))
        return output_lst

    def model_forward_step(
        self,
        *,
        inputs: torch.Tensor,
        extra_inputs: dict[str, torch.Tensor] | None = None,
        extra_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the trainer model without backward.
        """
        model_parts = self.module
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not yet supported in model_forward_step. "
                "This will be implemented in a follow-up PR."
            )
        else:
            # Non-PP forward. train_context (SPMD mesh) is set by the caller.
            assert len(model_parts) == 1
            folded = getattr(model_parts[0], "folded_token_stream", False)
            if folded and inputs.dim() == 2 and inputs.shape[0] == 1:
                # Folded-stream models (kimi_k3) take [T] token streams; the
                # rmpad path packs to [1, T]. Fold in, unfold the logits out.
                squeezed_inputs = inputs.squeeze(0)
                squeezed_extra = {
                    k: (v.squeeze(0) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 else v)
                    for k, v in (extra_inputs or {}).items()
                }
                pred = model_parts[0](squeezed_inputs, **squeezed_extra, **extra_kwargs)
                if pred.dim() == 2:
                    pred = pred.unsqueeze(0)
            else:
                pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)

        return self._finish_pred(pred)

    def _finish_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Bring a stage's logits to the layout the loss side expects."""
        parallel_dims = self.parallel_dims
        if isinstance(pred, DTensor):
            pred = pred.full_tensor()
        if parallel_dims.cp_enabled:
            # Inputs were seq-sharded across cp; the loss side works on
            # full sequences (see prepare_model_inputs), so gather the
            # logits back (differentiable -> reduce-scatter backward).
            cp_group = parallel_dims.get_mesh("cp").get_group()
            pred = gather_outputs_and_unpad(
                pred.contiguous(), gather_dim=1, group=cp_group
            )
        return pred

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.module for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            logger.warning(f"grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def lr_scheduler_step(self):
        """Advance learning rate scheduler."""
        self.lr_scheduler.step()
        lr = self.lr_scheduler.schedulers[0].get_last_lr()[0]
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Move model and/or optimizer to CPU or GPU."""
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            return

        device_name = get_device_name()
        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                for module in self.module:
                    load_fsdp_model_to_gpu(module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                for module in self.module:
                    offload_fsdp_model_to_cpu(module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save checkpoint."""
        if self._is_offload_param:
            for module in self.module:
                load_fsdp_model_to_gpu(module)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        if max_ckpt_to_keep is not None:
            self.checkpointer.keep_latest_k = max_ckpt_to_keep

        self.checkpointer.save(curr_step=global_step)

        torch.distributed.barrier()
        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """Load checkpoint."""
        if self._is_offload_param:
            for module in self.module:
                load_fsdp_model_to_gpu(module)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        # Extract step number from path (verl uses global_step_N format)
        match = re.search(r"global_step_(\d+)", local_path)
        if match:
            step = int(match.group(1))
            self.checkpointer.load(step=step)
        else:
            # Fallback to latest
            self.checkpointer.load(step=-1)

        torch.distributed.barrier()
        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(self, base_sync_done: bool = False, **kwargs):
        for module in self.module:
            load_fsdp_model_to_gpu(module)

        # Adapter-mode sync, the answer to the old "support Torchtitan PEFT" TODO.
        # engine_workers gates its base-then-adapter sequence on peft_config being
        # present, so returning None forced every LoRA run through a merged full-weight
        # sync even when it asked for model.lora.merge=False -- LoRA then bought
        # optimizer and gradient memory but none of its sync bandwidth.
        wrappers = {}
        for module in self.module:
            wrappers.update(_titan_lora_wrappers(module))
        # Adapter mode is OPT-IN. model_config.lora defaults to an EMPTY dict, so
        # reading merge off it with a False default would flip every existing LoRA run
        # onto this path -- and the merged path is the one with end-to-end evidence.
        # An empty (or absent) lora block means the run said nothing about PEFT, so it
        # stays merged; a configured block honours its own merge flag, defaulting to
        # False the way the megatron engine does.
        lora_cfg = getattr(self.model_config, "lora", None)
        if not isinstance(lora_cfg, dict):
            lora_cfg = {} if lora_cfg is None else dict(vars(lora_cfg))
        peft_merge = bool(lora_cfg.get("merge", False)) if lora_cfg else True
        adapter_mode = bool(wrappers) and not peft_merge
        peft_config = _peft_config_from_wrappers(wrappers) if adapter_mode else None

        params = {}
        merged_lora_keys: set[str] = set()
        if adapter_mode:
            # Unmerged: to_hf renames <fqn>.base.weight to <fqn>.weight and drops the
            # adapter tensors, which is exactly the base half of the sequence.
            for module in self.module:
                params.update(module.state_dict())
            # Point the checksum probe at the adapters: they are the only tensors that
            # move under LoRA, and the base half is frozen by construction. lora_b
            # starts at zero, so lora_a is listed first -- a zero digest on step 1 is
            # correct but reads like a broken probe.
            merged_lora_keys = {
                k for k in params if k.endswith(("lora_a", "lora_b"))
            }
        else:
            for module in self.module:
                module_params, module_merged = _merged_state_dict_if_lora(module)
                params.update(module_params)
                merged_lora_keys.update(module_merged)

        # DIAGNOSTIC, off unless KIMI_GRPO_FREEZE_SYNC=1. Ships the FIRST step's
        # weights forever, so the rollout engine runs a stale policy while the actor
        # keeps training.
        #
        # It exists to make rollout_probs_diff interpretable. With a zero-variance
        # reward the actor never moves, so a working sync and a no-op sync give
        # identical probs_diff and the metric proves nothing -- measured:
        # grad_norm was exactly 0.0. With a variance reward the actor does move and
        # probs_diff grows 1.75e-03 -> 2.45e-03 over three steps, but that alone
        # does not show the metric is SENSITIVE to the sync. Freezing the sync is
        # the differential: if probs_diff does not grow well past the synced
        # baseline, then no value of it says anything about the sync.
        import os as _os

        if _os.environ.get("KIMI_GRPO_FREEZE_SYNC") == "1":
            if not hasattr(self, "_frozen_sync_params"):
                self._frozen_sync_params = {
                    k: v.detach().clone() for k, v in params.items()
                }
                logger.warning(
                    "KIMI_GRPO_FREEZE_SYNC=1: caching %d tensors and shipping them "
                    "to the rollout engine for every later step. DIAGNOSTIC ONLY -- "
                    "the rollout policy is deliberately stale.",
                    len(params),
                )
            params = dict(self._frozen_sync_params)

        # Direct check instead of inferring from rollout_probs_diff. A checksum of
        # one real parameter, logged every sync: it must CHANGE across steps when the
        # sync is live and stay CONSTANT when frozen. Inferring from probs_diff
        # failed because the two arms sample different responses, so the metric is
        # not comparing the same tokens -- an uncontrolled comparison, not evidence.
        if _os.environ.get("KIMI_GRPO_SYNC_CHECKSUM", "1") == "1":
            import hashlib as _hl

            # Prefer a key the LoRA merge produced: under LoRA everything else is
            # frozen, so any other choice gives a constant digest and says nothing.
            probe = None
            for keyset in (sorted(merged_lora_keys), sorted(params)):
                for k in keyset:
                    v = params.get(k)
                    if (
                        v is not None
                        and hasattr(v, "numel")
                        and v.numel() > 1
                        and v.dtype.is_floating_point
                    ):
                        probe = (k, v)
                        break
                if probe is not None:
                    break
            if probe is not None:
                k, v = probe
                t = v.to_local() if hasattr(v, "to_local") else v
                digest = _hl.sha256(
                    t.detach().float().cpu().contiguous().numpy().tobytes()
                ).hexdigest()[:16]
                logger.warning("SYNC-CHECKSUM %s %s", digest, k)

        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

        # Convert TorchTitan key names to HuggingFace key names (expected by vLLM)
        sd_adapter = self.checkpointer.sd_adapter
        hf_names = {}
        if sd_adapter is not None:
            if adapter_mode:
                hf_names = _wrapped_hf_base_names(sd_adapter, wrappers, params)
            params = sd_adapter.to_hf(params)
        elif adapter_mode:
            raise ValueError(
                "adapter-only weight sync needs the state-dict adapter to name the "
                "wrapped projections; none is configured"
            )

        if adapter_mode:
            if base_sync_done:
                # Second half of the sequence: only what LoRA learned goes over.
                params = _adapter_state_dict(wrappers, hf_names)
            else:
                params = _insert_base_layer_suffix(
                    params, getattr(self.model_config.hf_config, "model_type", "")
                )
            logger.warning(
                "weight sync: adapter mode, base_sync_done=%s, shipping %d tensors "
                "(rank %d, %d wrapped projections)",
                base_sync_done,
                len(params),
                peft_config["r"],
                len(wrappers),
            )

        # When weight tying is enabled, the sd_adapter skips lm_head.weight during
        # to_hf() conversion (since it's the same tensor as embed_tokens.weight in
        # the torchtitan model). But vLLM needs lm_head.weight explicitly, so we
        # add it back as a reference to embed_tokens.weight.
        if "model.embed_tokens.weight" in params and "lm_head.weight" not in params:
            params["lm_head.weight"] = params["model.embed_tokens.weight"]

        device = get_device_id()  # used when fsdp2 set cpu_offload_policy

        # When Expert Parallel (EP) is used, sd_adapter.to_hf() only produces
        # individual expert weights for the locally-owned experts (e.g., 16 out of
        # 128 with EP=8). vLLM needs ALL experts. We gather the missing experts
        # by all-gathering each expert weight across the EP process group.
        # The adapter half carries no expert tensors -- routed experts are 3-D
        # GroupedExperts parameters and cannot be LoRA-wrapped -- so there is nothing
        # for the EP gather to complete, and it is skipped rather than handed a dict it
        # would find no expert keys in. The BASE half is the full model and still
        # gathers.
        if self.parallel_dims.ep_enabled and not (adapter_mode and base_sync_done):
            ep_mesh = self.parallel_dims.get_optional_mesh("ep")
            ep_group = ep_mesh.get_group()
            ep_size = self.parallel_dims.ep
            per_tensor_param = iter_per_tensor_params_ep(params, device, ep_group, ep_size)
        else:
            # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
            per_tensor_param = (
                (
                    name,
                    param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                    if isinstance(param, DTensor)
                    else param,
                )
                for name, param in params.items()
            )
        return per_tensor_param, peft_config


def _titan_lora_wrappers(module):
    """``{fqn: wrapper}`` for every KimiLoRALinear in the module.

    Discovered from the module rather than from the config, because ``apply_lora``
    decides what actually got wrapped (its target list matches leaf names AND
    qualified suffixes, and it skips the KDA subtree structurally). A config-derived
    list would claim targets that were never wrapped.
    """
    found = {}
    for name, sub in module.named_modules():
        if (
            hasattr(sub, "lora_a")
            and hasattr(sub, "lora_b")
            and hasattr(sub, "base")
        ):
            found[name] = sub
    return found


def _peft_config_from_wrappers(wrappers):
    """A vLLM PEFTHelper-compatible dict, or None when there is nothing to describe.

    ``target_modules`` is the set of leaf names actually wrapped. Our LoRA targets are
    already HF-style leaf names (``q_proj``, ``o_proj``, ``gate_proj``, ...), so unlike
    the megatron path there is no megatron-to-HF target rename to do.

    rank and alpha come off a wrapper instead of the config: the wrapper stores
    ``alpha / rank`` as ``_lora_scaling`` and its shapes carry the rank, so the values
    reported are the ones the adapters were actually built with.
    """
    if not wrappers:
        return None
    from peft import TaskType

    any_wrapper = next(iter(wrappers.values()))
    rank = int(any_wrapper.lora_a.shape[0])
    alpha = float(getattr(any_wrapper, "_lora_scaling", 1.0)) * rank
    ranks = {int(w.lora_a.shape[0]) for w in wrappers.values()}
    if len(ranks) > 1:
        raise ValueError(
            f"adapter-only sync needs one rank for all wrappers, got {sorted(ranks)}"
        )
    return {
        "task_type": TaskType.CAUSAL_LM,
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": sorted({fqn.rsplit(".", 1)[-1] for fqn in wrappers}),
        "exclude_modules": [],
        "bias": "none",
        "lora_dropout": 0.0,
    }


def _wrapped_hf_base_names(sd_adapter, wrappers, full_state_dict):
    """``{fqn: hf_name}`` for each wrapped projection's BASE weight.

    Uses the adapter's own key mapping rather than reimplementing it -- the vision /
    text prefixing, the official-export renames and the ``.base.weight`` stripping all
    live there, and a second copy of that logic is how the two drift apart. ``to_hf``
    decides text-vs-multimodal from the WHOLE state dict, so ``_is_text_only`` is asked
    once against the full dict; calling ``to_hf`` per key would let a one-entry dict
    misclassify it and silently emit the wrong prefix.

    ``to_hf`` itself cannot carry the adapters: it drops every ``lora_a`` / ``lora_b``
    key by design, because the HF key space is the original Kimi architecture.
    """
    from torchtitan.models.kimi_k3.lora import _state_dict_prefix

    text_only = sd_adapter._is_text_only(full_state_dict)
    # The fqns come from named_modules(), which KEEPS wrapper segments that state_dict()
    # strips: under activation checkpointing `layers.0.ffn.gate_proj` is
    # `layers.0._checkpoint_wrapped_module.ffn.gate_proj` there. Composing a key from the
    # module path then hands to_hf a name it has no mapping for --
    #   ValueError: Unmapped tt key:
    #   'layers.0._checkpoint_wrapped_module.ffn.gate_proj.weight'
    # which is exactly the failure merge_lora_state_dict already hit from the same
    # direction. _state_dict_prefix is that fix; reusing it keeps one source of truth and
    # makes it VALIDATE against the state dict instead of guessing a stripped name.
    return {
        fqn: sd_adapter._tt_key_to_hf(
            f"{_state_dict_prefix(fqn, full_state_dict)}.weight", text_only
        )
        for fqn in wrappers
    }


def _adapter_state_dict(wrappers, hf_names):
    """PEFT-named adapter tensors for the wrapped projections.

    Exported UNSCALED: PEFT applies ``lora_alpha / r`` from the config it is handed, so
    pre-multiplying by the wrapper's ``_lora_scaling`` would apply the scale twice.

    Safe to ship the raw factors because the base mapping this borrows names from is a
    pure RENAME for every LoRA target. The only value transform on the single-tensor
    path is the 4-D ``A_log`` reshape, and ``apply_lora`` skips the KDA subtree
    structurally, so no wrapped module is ever reshaped. Routed experts are likewise
    out of reach -- they are 3-D GroupedExperts parameters, not ``nn.Linear``.
    """
    out = {}
    for fqn, wrapper in wrappers.items():
        stem = hf_names[fqn].removesuffix(".weight")
        out[f"{stem}.lora_A.weight"] = wrapper.lora_a
        out[f"{stem}.lora_B.weight"] = wrapper.lora_b
    return out


# vLLM wraps EVERY linear layer when LoRA is enabled -- `get_supported_lora_modules`
# collects leaf names by module TYPE, not from the adapter's target_modules -- so the
# base half has to name a projection `base_layer` whenever the ROLLOUT wrapped it, which
# is a wider set than the one torchtitan wrapped. K3 diverges on the KDA subtree, which
# `apply_lora` skips structurally while vLLM wraps it anyway.
#
# The set is decided by one question per key: is its vLLM destination a LinearBase
# subclass? Not "does the name look like a projection" -- the KDA short conv is named
# conv1d and is a ColumnParallelLinear, while embeddings and lm_head are linear-shaped and
# are NOT wrapped, because K3 declares no embedding_modules. Norms, A_log and dt_bias are
# parameters of non-linear modules and stay plain.
#
# These are the K3 linear projections absent from verl's shared STACKED_PARAMS. The
# rename applies to the SOURCE name and vLLM's stacked mapping is a substring replace,
# so `.q_proj.base_layer.weight` still resolves to `.in_proj_qkvgfab.base_layer.weight`.
# Getting one wrong is not silent: the destination misses params_dict, vLLM's stacked
# loop treats that as "packed projection not present on this layer" and falls through to
# the plain path, and the KeyError there names the projection.
_K3_STACKED_PARAMS = (
    # KDA, fused into in_proj_qkvgfab (q/k/v/b/f_a/g) plus its own f_b_proj
    ".b_proj.weight",
    ".f_a_proj.weight",
    ".f_b_proj.weight",
    ".g_proj.weight",
    # KDA short convolution. vLLM models it AS a linear -- `self.conv1d =
    # ColumnParallelLinear(...)`, unsqueezed to conv layout afterwards -- so it is
    # LoRA-wrapped like any projection. Reading the name is not enough here.
    ".q_conv1d.weight",
    ".k_conv1d.weight",
    ".v_conv1d.weight",
    # Block AttnRes graft
    ".self_attention_res_proj.weight",
    ".output_attn_res_proj.weight",
    ".mlp_res_proj.weight",
    # Latent MoE shared W_down / W_up (report Eq. 11), ReplicatedLinear in vLLM.
    ".routed_expert_down_proj.weight",
    ".routed_expert_up_proj.weight",
    # The per-expert w1/w2/w3 are deliberately NOT here. RoutedExperts is a
    # PluggableLayer, not a linear, so LoRA never wraps it and its params keep their plain
    # names. vLLM's expert mapping did demand the base_layer form, but from a GLOBAL
    # `any(".base_layer." in n for n in model.named_parameters())` probe -- true as soon as
    # any projection is wrapped -- which made it ask for a destination that does not exist.
    # Fixed on the vLLM side by scoping that probe to the routed-expert params.
    # MoE router: GateLinear(ReplicatedLinear). STACKED_PARAMS spells this `.mlp.gate.`,
    # which K3 does not use.
    ".block_sparse_moe.gate.weight",
    ".block_sparse_moe.gate.e_score_correction_bias",
)


def _insert_base_layer_suffix(params, model_type):
    """Rename each LoRA-wrappable projection's HF base key to PEFT's ``base_layer`` form.

    The base sync ships the FULL model -- embeddings, norms, experts, everything -- and
    only the projections the rollout wraps change name.
    """
    return dict(
        add_base_layer_suffix(
            params.items(),
            model_type=model_type,
            extra_stacked_params=_K3_STACKED_PARAMS,
        )
    )


def _merged_state_dict_if_lora(module):
    """``module.state_dict()``, with LoRA adapters folded into the base weights.

    A LoRA-wrapped projection stores ``base.weight``, ``lora_a`` and ``lora_b``. The
    state-dict adapter maps ``base.weight`` to the plain HF name and has no mapping
    for the adapter tensors, so a raw state_dict ships the UNMERGED base and silently
    drops everything LoRA learned. Under LoRA the base is frozen, so the rollout
    engine would then receive the same weights at every step -- indistinguishable
    from a broken sync, and the actor would train adapters the rollout never sees.

    Measured on kimi_k3_debugmodel_gated_lora: with lora_b at its zero init the two
    paths agree (LoRA is identity at step 0, so that is correct); with lora_b set to
    0.01 the merged path changes and the raw path does not. Key sets are identical
    either way, 151 both, so this is a drop-in for the sync.

    Non-LoRA models take the plain path -- the import and the scan are both skipped
    unless a wrapper is actually present.
    """
    has_lora = any(
        hasattr(m, "lora_a") and hasattr(m, "lora_b") and hasattr(m, "base")
        for m in module.modules()
    )
    if not has_lora:
        return module.state_dict(), frozenset()
    from torchtitan.models.kimi_k3.lora import merge_lora_state_dict

    # Merged because the run asked for it (model.lora.merge=True, the default). A run
    # with merge=False takes the adapter-only path in get_per_tensor_param instead and
    # never reaches here.
    # warning, not info: the first run of this shipped with logger.info and the line
    # never appeared, so "the merge ran" was inferred rather than read. Engagement has
    # to be assertable from the log.
    logger.warning(
        "weight sync: folding LoRA adapters into base weights before to_hf; "
        "shipping the raw state dict would send the frozen base only. Set "
        "model.lora.merge=False for the adapter-only sync instead."
    )
    merged = merge_lora_state_dict(module)
    # Which keys the merge produced, so the checksum probe can pick one of THEM. It
    # otherwise takes the first floating-point key in sorted order, which is
    # embed_tokens.weight -- frozen under LoRA, so its checksum is constant whether the
    # sync works or not. Measured: four syncs, four identical digests, proving nothing.
    raw = set(module.state_dict())
    return merged, frozenset(k for k in merged if k not in raw)


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: TorchTitanEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanEngine)
        super().__enter__()
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanEngine)

        # Reshard the root FSDP module
        if self.engine.engine_config.data_parallel_shard_size > 1:
            for module in self.engine.module:
                module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: TorchTitanEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanEngine)
        super().__enter__()
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanEngine)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["torchtitan"], device=["cuda", "npu"])
class TorchTitanEngineWithLMHead(TorchTitanEngine):
    """TorchTitan engine implementation for language models with LM head."""

    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        output_args = {}

        if use_remove_padding:
            input_ids = input_ids.values().unsqueeze(0)
            if position_ids.dim() == 3:
                position_ids = position_ids.values().unsqueeze(1)
            else:
                position_ids = position_ids.values().unsqueeze(0)

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            attn_type = self.engine_config.attn_type
            attention_mask = get_attention_masks(
                input_batch=input_ids,
                positions=position_ids,
                attn_type=attn_type,
            )
        else:
            loss_mask = micro_batch["loss_mask"]
            pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            batch_size = micro_batch.batch_size[0]
            max_seq_len = max(input_ids.offsets().diff())

            labels = torch.roll(input_ids.values(), shifts=-1, dims=0)
            input_ids = torch.nested.to_padded_tensor(
                input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
            )

            if position_ids.dim() == 3:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                ).transpose(0, 1)
            else:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, max_seq_len)
                )

            attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
            attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
            attention_mask = torch.nested.to_padded_tensor(
                attention_mask, padding=0, output_size=(batch_size, max_seq_len)
            )

        extra_inputs = {
            "positions": position_ids,
        }
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {"attention_masks": attention_mask}
        cp_pad_len = 0
        if self.parallel_dims.cp_enabled:
            # Context parallel wants a sequence it can cut evenly, and the flex
            # BlockMask wants each shard to be a whole number of 128-token
            # blocks; a packed no-padding stream is neither. ulysses_pad does
            # the padding, with two adjustments this engine needs.
            #
            # The multiple: ulysses_pad pads to the parallel degree, which is
            # what a model whose CP lives inside its own modules needs; a model
            # whose CP goes through the flex BlockMask needs whole blocks per
            # shard, so the modulus handed to it is cp * 2 * 128 there.
            cp_size = self.parallel_dims.cp
            multiple = cp_size if self._model_cp_is_module_internal else cp_size * 2 * 128
            pad_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            input_ids, position_ids, cp_pad_len = ulysses_pad(
                input_ids, position_ids, sp_size=multiple, pad_value=pad_id
            )
            if cp_pad_len:
                # The positions: ulysses_pad numbers the padding from zero,
                # which reads as the start of a new document to a mask built
                # from positions -- this model's does. Renumber the padding to
                # continue the stream instead.
                continued = torch.arange(
                    1, cp_pad_len + 1, device=position_ids.device
                )
                if position_ids.dim() == 3:
                    position_ids[..., -cp_pad_len:] = (
                        position_ids[..., -cp_pad_len - 1 : -cp_pad_len] + continued
                    )
                else:
                    position_ids[:, -cp_pad_len:] = (
                        position_ids[:, -cp_pad_len - 1 : -cp_pad_len] + continued
                    )
                # Labels ride along so they stay aligned with the logits; both
                # are unpadded before the loss (prepare_model_outputs).
                labels = torch.nn.functional.pad(labels, (0, cp_pad_len), value=pad_id)
                extra_inputs["positions"] = position_ids
                if attention_mask is not None:
                    attention_mask = get_attention_masks(
                        input_batch=input_ids,
                        positions=position_ids,
                        attn_type=self.engine_config.attn_type,
                    )
                    extra_kwargs["attention_masks"] = attention_mask
            # prepare_context_parallel_input contract: positions must ride
            # in extra_kwargs (it seq-shards them alongside inputs/labels);
            # this engine keeps positions in extra_inputs, so bridge them
            # across the call. Was a latent KeyError -- this CP path had
            # never been exercised before the kimi_k3 CP work.
            extra_kwargs["positions"] = extra_inputs["positions"]
            # Module-internal-CP models (kimi_k3) are causal-only and never
            # consume attention_masks; upstream's BlockMask CP sharding also
            # requires seq_len % (cp * 128) == 0, which SFT batches don't
            # guarantee. Keep the mask out of the CP shard for them.
            masks = (
                extra_kwargs.pop("attention_masks", None)
                if self._model_cp_is_module_internal
                else None
            )
            # Keep labels FULL length: verl's loss path (nested no-padding
            # log_prob/loss_mask handling) assumes full sequences, so the
            # engine all-gathers the seq-sharded logits after the model
            # call (model_forward_step) instead of sharding the loss side.
            labels_full = labels
            # cp_shard cuts dim 0 and this stream is [1, T] -- dim 0 is the
            # fold, not the sequence, so sharding it hands every rank but the
            # first an empty batch. Fold the stream down to [T] for the cut
            # (the model's own contract anyway) and unfold after.
            folded_for_cp = input_ids.dim() == 2 and input_ids.shape[0] == 1
            if folded_for_cp:
                input_ids = input_ids.squeeze(0)
                labels = labels.squeeze(0)
                positions = extra_kwargs["positions"]
                extra_kwargs["positions"] = (
                    positions.squeeze(0) if positions.dim() > 1 and positions.shape[0] == 1
                    else positions
                )
            input_ids, _labels_sharded, extra_kwargs = prepare_context_parallel_input(
                input_ids,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.trainer.device,
                # NO_PADDING packs variable-length sequences, so the
                # head-tail balancer's seq % (2*cp) == 0 precondition cannot
                # hold; shard contiguously.
                None,
            )
            labels = labels_full
            extra_inputs["positions"] = extra_kwargs.pop("positions")
            if folded_for_cp:
                # Back to the rmpad layout the rest of the engine expects.
                input_ids = input_ids.unsqueeze(0)
                if extra_inputs["positions"].dim() == 1:
                    extra_inputs["positions"] = extra_inputs["positions"].unsqueeze(0)
            if masks is not None:
                extra_kwargs["attention_masks"] = masks

        # TODO(jessicazhong): multimodal is not yet supported for Torchtitan engine
        extra_inputs.update(multi_modal_inputs)
        output_args["labels"] = labels
        output_args["cp_pad_len"] = cp_pad_len
        return input_ids, extra_inputs, extra_kwargs, output_args

    def prepare_model_outputs(self, logits, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        temperature = micro_batch["temperature"]
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        labels = output_args["labels"]
        # Drop the rows the CP padding added, so everything below sees the
        # packed stream at its true length (see prepare_model_inputs).
        cp_pad_len = output_args.get("cp_pad_len", 0)
        if cp_pad_len:
            logits = logits[:, :-cp_pad_len]
            labels = labels[:, :-cp_pad_len]
        model_output = {}

        input_ids = micro_batch["input_ids"]
        cu_seqlens = input_ids.offsets()
        if use_remove_padding:
            labels = labels.squeeze(0)
            logits_rmpad = logits.squeeze(0)
            # PyTorch's autograd doesn't allow in-place modification of views when gradients need to flow back
            logits_rmpad = logits_rmpad / temperature

            inplace_backward = True
            if calculate_entropy:
                inplace_backward = False
            log_probs = logprobs_from_logits(
                logits=logits_rmpad,
                labels=labels,
                inplace_backward=inplace_backward,
            )

            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    if self.engine_config.entropy_from_logits_with_chunking:
                        entropy_rmpad = self.compute_entropy_from_logits(
                            logits_rmpad,
                            chunk_size=self.engine_config.entropy_from_logits_chunk_size,
                        )  # ((total_nnz / sp) + pad)
                    else:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                else:
                    entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

            log_probs = torch.nested.nested_tensor_from_jagged(log_probs.squeeze(0), cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
        else:
            logits.div_(temperature)
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            seq_lengths = cu_seqlens.diff()
            starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
            logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            log_probs = logprobs_from_logits(logits=logits_rmpad, labels=output_args["labels"])
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            logits = self.model_forward_step(inputs=input_ids, extra_inputs=extra_inputs, extra_kwargs=extra_kwargs)

            model_output = self.prepare_model_outputs(logits=logits, output_args=output_args, micro_batch=micro_batch)

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output
