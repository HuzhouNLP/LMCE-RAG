import argparse
import json
import logging
import os
import pathlib
import re
from functools import partial
from itertools import chain
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.theme import Theme
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

from lkif.kb_encoder import LKIFEncoder
from lkif.models.kblam_config import LKIFConfig
from lkif.models.llama3_model import LkifLlamaForCausalLM
from lkif.models.phi3_model import LKIFPhi3ForCausalLM
from kblam.utils.data_utils import augment_row, generate_multi_entity_qa, get_i_dont_know_ans
from kblam.utils.train_utils import context_set_size_scheduler, get_kb_embd, setup_scheduler_and_optimizer

LOGFORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGFORMAT_RICH = "%(message)s"

# setup logging
# Create a custom theme for Rich
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }
)

# Create a Rich console with the custom theme
console = Console(theme=custom_theme)

# Configure the root logger to WARNING
logging.basicConfig(
    level=logging.WARNING,  # Set the root logger to WARNING
    format=LOGFORMAT_RICH,
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--train_dataset",type=str,default="QA-summary-ref")
parser.add_argument("--N", type=int, default=120000, help="Size of training set, select the first N samples for training")
parser.add_argument("--B", type=int, default=10, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--sep_query_head", action=argparse.BooleanOptionalAction, help="Train a separate query head")
parser.add_argument("--use_oai_embd", action="store_true", help="Use OpenAI embedding")
parser.add_argument("--use_cached_embd", action="store_true", help="Choose to use pre-computed KV embeddings")
parser.add_argument("--total_steps", type=int, default=20000, help="Total steps")
parser.add_argument("--encoder_spec", type=str, default="OAI")
parser.add_argument("--key_embd_src", type=str, default="key", choices=["key", "answer", "questions", None], help="Source of key embedding")
parser.add_argument("--use_data_aug", action="store_true", help="Randomly pick templates for the question")
parser.add_argument("--use_lr_decay", action="store_true")
parser.add_argument("--dataset_dir", type=str, default="date-synthesis/qar_generation/results")
parser.add_argument("--model_dir_to_resume", type=str, default=None, help="Checkpoint directory to resume training")
parser.add_argument("--hf_model_spec", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("--hf_token", type=str,default=None,help="Huggingface token")
parser.add_argument("--model_save_dir", type=str, default="output", help="Place to save the checkpoints")
parser.add_argument("--kb_size", type=int, default=None, help="The size of the KB set size")
parser.add_argument("--dynamic_kb_size", nargs=2, type=int, default=None, help="The size of the KB set size. Set a dynamic range for the kbsize specify min and max")
parser.add_argument("--duplicate_true_kb", action=argparse.BooleanOptionalAction, default=True, help="Duplicate true entity's KB token")
parser.add_argument("--length_invariance", action=argparse.BooleanOptionalAction, default=False, help="Scale the raw attention score")
parser.add_argument("--outlier_num", type=int, default=1, help="Introduce questions without correct KB entites")
parser.add_argument("--multi_entities", type=int, default=None, help="Introduce questions involving multiple entities")
parser.add_argument("--use_extended_qa", action="store_true", help="Introduce QA with extended open-ended parts")
parser.add_argument("--kb_token_layer_frequency", type=int, default=3, help="Introduce QA with extended open-ended parts")
parser.add_argument("--gradient_accm_step", type=int, default=20, help="Introduce QA with extended open-ended parts")
parser.add_argument("--verbose", action="store_true", help="Set logging to debug")
parser.add_argument("--log_to_file", action="store_true", help="Log to file as well as stdout")
parser.add_argument("--llm_type",type=str,default="llama3",choices=["llama3", "phi3"])
parser.add_argument("--max_seq_len",type=int,default=None)
parser.add_argument("--dataset_type", type=str, default="law", choices=["game", "law"], help="Type of dataset to process")
# fmt: on


def create_custom_progress_bar(
    console: Console = None,  # type: ignore
    color: str = "cyan",
    show_time: bool = True,
    show_spinner: bool = True,
    spinner_style: str = "dots",
    disable=False,
) -> Progress:
    """
    Create a custom progress bar using Rich, optionally including loss reporting.
    """
    if console is None:
        console = Console()
    columns = []

    if show_spinner:
        columns.append(SpinnerColumn(spinner_name=spinner_style, style=color))

    columns.extend(
        [
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None, style=color, complete_style=f"bold {color}"),
            TaskProgressColumn(),
            TextColumn("[bold yellow]Loss: {task.fields[loss]:.4f}", justify="right"),
        ]
    )

    if show_time:
        columns.append(TimeRemainingColumn())

    progress = Progress(*columns, console=console, expand=True, disable=disable)
    return progress


def _format_QA_llama(Q: str, A: str):
    return (
        "<|start_header_id|>user<|end_header_id|> "
        + Q
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>"
        + A
        + "<|eot_id|>"
    )


def _format_QA_phi3(Q: str, A: str):
    return "<|user|>\n" + Q + "<|end|>\n" + "<|assistant|>\n" + A + "<|end|>\n"


def _create_labels_for_llama(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
    answer_indices = torch.argmax(
        (input_ids == tokenizer("<|start_header_id|>assistant<|end_header_id|>")["input_ids"][1]).long(),
        -1,
    )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 2)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels


def _create_labels_for_phi3(input_ids: torch.Tensor, input_strs: List[str], tokenizer):
    answer_indices = torch.argmax(
        (input_ids == tokenizer("<|user|>")["input_ids"][0]).long(),
        -1,
    )
    answer_mask = torch.ones_like(input_ids)
    for b in range(len(input_strs)):
        answer_mask[b, : (answer_indices[b].item() + 1)] = 0
    labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
    return labels


def get_batch(
    qa_format_func: Callable[[str, str], str],
    label_func: Callable[[torch.Tensor, List, Callable], torch.Tensor],
    dataset: List[Dict],
    tokenizer,
    device: torch.device,
    B: int = 20,
    random_sample=True,
    use_data_aug=False,
    include_outlier=False,
    multi_entities=None,
    use_extended_qa=False,
    dataset_type="law",
):
    """
    适配不同数据集类型的batch生成函数
    """
    labels = []
    if multi_entities is not None:
        assert not include_outlier

    if random_sample:
        if multi_entities is not None:
            batch_indices = np.random.choice(len(dataset), (B, multi_entities), replace=False)
        else:
            batch_indices = np.random.choice(len(dataset), B, replace=False)
    else:
        batch_indices = np.arange(B)

    def get_question_and_answer(idx: int) -> tuple[str, str]:
        if use_extended_qa:
            Q, A = dataset[idx]["extended_Q"], dataset[idx]["extended_A"]

        elif multi_entities is not None:
            # 适配多实体处理，根据数据集类型提取不同字段
            if dataset_type == "law":
                names = [dataset[i]["domain"] for i in idx]
                desc_types = [dataset[i]["language"] for i in idx]
                descriptions = [dataset[i]["ground_truth"]["content"] for i in idx]
            else:  # game
                names = [dataset[i]["name"] for i in idx]
                desc_types = [dataset[i]["description_type"] for i in idx]
                descriptions = [dataset[i]["description"] for i in idx]
                
            Q, A = generate_multi_entity_qa(names, desc_types, descriptions)
        else:
            # 根据数据集类型提取问题和答案
            if dataset_type == "law":
                Q = augment_row(dataset[idx], dataset_type) if use_data_aug else dataset[idx]["query"]["content"]
                A = get_i_dont_know_ans() if include_outlier else dataset[idx]["ground_truth"]["content"]
            else:  # game
                Q = augment_row(dataset[idx], dataset_type) if use_data_aug else dataset[idx]["Q"]
                A = get_i_dont_know_ans() if include_outlier else dataset[idx]["A"]
        return Q, A

    with torch.autograd.no_grad():
        input_strs = []
        real_batch_indices = []
        for idx in batch_indices:
            Q, A = get_question_and_answer(idx)
            if Q is not None and A is not None:
                input_strs.append(qa_format_func(Q, A))
                real_batch_indices.append(idx)
            else:
                print("Q or Answer is none")
        batch_indices = real_batch_indices
        tokenizer_output = tokenizer(input_strs, return_tensors="pt", padding=True).to(device)
        input_ids, attention_masks = (
            tokenizer_output["input_ids"],
            tokenizer_output["attention_mask"],
        )

        labels = label_func(input_ids, input_strs, tokenizer)
    if include_outlier:
        batch_indices = np.random.choice(len(dataset), B, replace=False)
    return input_ids, attention_masks, labels, batch_indices


def get_prefix_str(args):
    use_data_aug = args.use_data_aug
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size

    if dynamic_kb_size is not None:
        kb_size = "dynamic"  # Random size

    duplicate_true_kb = args.duplicate_true_kb
    length_invariance = args.length_invariance
    outlier_ratio = args.outlier_num
    use_outlier = outlier_ratio != -1
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    lr = args.lr
    dataset_type = args.dataset_type

    prefix_string = f"stage1_lr_{lr}_ds_{dataset_type}"
    if kb_token_layer_frequency is not None:
        prefix_string += f"KBTokenLayerFreq{kb_token_layer_frequency}"
    if use_extended_qa:
        prefix_string += "UseExtendedQA"
    if multi_entities is not None:
        prefix_string += f"MultiEntities{multi_entities}"
    if use_outlier:
        prefix_string += f"UseOutlier{outlier_ratio}"
    if length_invariance:
        prefix_string += "LengthInvariant"
    if not duplicate_true_kb:
        prefix_string += "NoDuplicate"
    if kb_size is not None:
        prefix_string += f"KBSize{kb_size}"
    if sep_query_head:
        prefix_string += "SepQueryHead"
    if use_data_aug:
        prefix_string += "UseDataAug"
    return prefix_string


def _load_cached_embeddings(encoder_model_spec: str, dataset_dir: str, dataset_name: str, key_embd_src: str):
    """
    加载缓存的嵌入，与encoder保持一致的命名规范
    """
    # 与encoder中save_name保持一致
    if encoder_model_spec == "OAI":
        encoder_model_spec_str = "OAI"
    elif encoder_model_spec == "ada-embeddings":
        encoder_model_spec_str = "OAI"
    elif encoder_model_spec == "text-embedding-3-large":
        encoder_model_spec_str = "BigOAI"
    elif encoder_model_spec == "all-MiniLM-L6-v2":
        encoder_model_spec_str = "all-MiniLM-L6-v2"
    elif encoder_model_spec == "bge-large-en-v1.5":
        encoder_model_spec_str = "bge-large-en-v1.5"
    else:
        encoder_model_spec_str = encoder_model_spec

    # 默认在 dataset_dir/embedding 下查找，如不存在再回退到 dataset_dir 根目录，兼容旧路径
    embedding_dir = os.path.join(dataset_dir, "embedding")

    # 加载key嵌入 (对应query.content)
    key_embd_path = os.path.join(
        embedding_dir,
        f"{dataset_name}_{encoder_model_spec_str}_embd_key.npy",
    )
    if not os.path.exists(key_embd_path):
        key_embd_path = os.path.join(
            dataset_dir,
            f"{dataset_name}_{encoder_model_spec_str}_embd_key.npy",
        )
    if not os.path.exists(key_embd_path):
        raise FileNotFoundError(f"Key embedding file not found: {key_embd_path}")
    key_embds = np.load(key_embd_path).astype("float32")

    # 加载value嵌入 (对应ground_truth.references)
    value_embd_path = os.path.join(
        embedding_dir,
        f"{dataset_name}_{encoder_model_spec_str}_embd_value.npy",
    )
    if not os.path.exists(value_embd_path):
        value_embd_path = os.path.join(
            dataset_dir,
            f"{dataset_name}_{encoder_model_spec_str}_embd_value.npy",
        )
    if not os.path.exists(value_embd_path):
        raise FileNotFoundError(f"Value embedding file not found: {value_embd_path}")
    value_embds = np.load(value_embd_path).astype("float32")

    return key_embds, value_embds


def get_step_config(
    current_accum_step: int,
    total_accum_step: int,
    use_data_aug: bool,
    outlier_num: int,
    multi_entities: int | None,
    use_extended_qa: bool,
):
    """
    控制不同类型QA的训练策略
    """
    config = {}
    config["use_data_aug"] = use_data_aug
    config["include_outlier"] = False
    config["multi_entities"] = None
    config["use_extended_qa"] = False
    include_outlier = current_accum_step >= total_accum_step - 1 - outlier_num
    if include_outlier:
        config["include_outlier"] = True
        return config
    if current_accum_step % 3 == 0:
        config["multi_entities"] = multi_entities
        return config
    if current_accum_step % 3 == 1:
        config["use_extended_qa"] = use_extended_qa
        return config
    return config


def _get_parameter_count(encoder):
    param_count = 0.0
    for p in encoder.parameters():
        if p.requires_grad:
            param_count += p.numel()
    return param_count


def _get_phi3_query_head_parameters(
    model: KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:
            if "qkv_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight[: model.config.hidden_size, :])  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params


def _get_llama3_query_head_parameters(
    model: KblamLlamaForCausalLM | KBLaMPhi3ForCausalLM,
    sep_query_head: bool,
    kb_token_layer_frequency: int,
):
    llm_q_params = []
    for name, param in model.named_parameters():
        if sep_query_head:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    old_weight = param.detach()
            if "q_proj_new.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.copy_(old_weight)  # type: ignore
                    param.requires_grad = True
                    llm_q_params.append(param)
        else:
            if "q_proj.weight" in name:
                layer_id = int(re.search(r"\d+", name)[0])  # type: ignore
                if layer_id % kb_token_layer_frequency == 0:
                    param.requires_grad = True
                    llm_q_params.append(param)
    return llm_q_params


class KBRetriever:
    def __init__(
        self,
        encoder: LKIFEncoder,
        dataset: List[Dict],
        key_embds: Optional[np.ndarray],
        value_embds: Optional[np.ndarray],
        dataset_type: str = "law",
    ):
        self.encoder = encoder
        self.key_embds = key_embds
        self.value_embds = value_embds
        self.dataset = dataset
        self.dataset_type = dataset_type
        
        # 预处理 ground_truth.references，兼容字符串 / 列表 / 字典
        self.gt_refs_str_list = []
        for data in dataset:
            if dataset_type == "law":
                refs = data.get("ground_truth", {}).get("references", [])

                # 1) 直接是字符串（例如完整或裁剪后的判决书）
                if isinstance(refs, str):
                    refs_str = refs

                # 2) 列表形式：可能是字符串列表或 dict 列表
                elif isinstance(refs, list) and refs:
                    if all(isinstance(ref, str) for ref in refs):
                        refs_str = "\n".join(refs)
                    elif all(isinstance(ref, dict) for ref in refs):
                        refs_str = "\n".join([ref.get("content", str(ref)) for ref in refs])
                    else:
                        refs_str = "\n".join([str(ref) for ref in refs])

                # 3) 其他情况（空或未知结构）
                else:
                    refs_str = ""

                self.gt_refs_str_list.append(refs_str)

            else:  # game
                self.gt_refs_str_list.append(data.get("description", ""))

    def _use_cached_embd(self):
        return self.key_embds is not None and self.value_embds is not None

    def get_key_embeddings(self, batch_indices, batch_size, step, kb_size):
        if self._use_cached_embd():
            # 使用缓存的嵌入
            train_set_key, train_set_val = get_kb_embd(
                self.encoder,
                batch_indices,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            # 实时生成嵌入，根据数据集类型提取不同字段
            if self.dataset_type == "law":
                # key: query.content, value: ground_truth.references
                batch_key_texts = [self.dataset[idx]["query"]["content"] for idx in batch_indices]
                batch_value_texts = [self.gt_refs_str_list[idx] for idx in batch_indices]
            else:  # game
                # key: Q, value: description
                batch_key_texts = [self.dataset[idx]["Q"] for idx in batch_indices]
                batch_value_texts = [self.dataset[idx]["description"] for idx in batch_indices]
                
            train_set_key = self.encoder.encode(batch_key_texts, convert_to_tensor=True)
            train_set_val = self.encoder.encode(batch_value_texts, convert_to_tensor=True)

        # 调整维度
        if len(train_set_key.shape) == 2:
            train_set_key = train_set_key.unsqueeze(0).transpose(0, 1)
            train_set_val = train_set_val.unsqueeze(0).transpose(0, 1)

        # 处理上下文集
        context_set_size = context_set_size_scheduler(step, kb_size)
        context_set_index = np.random.choice(len(self.dataset), context_set_size, replace=False)  # type: ignore
        
        if self._use_cached_embd():
            context_set_key, context_set_val = get_kb_embd(
                self.encoder,
                context_set_index,
                precomputed_embd=(self.key_embds, self.value_embds),
            )
        else:
            if self.dataset_type == "law":
                context_key_texts = [self.dataset[idx]["query"]["content"] for idx in context_set_index]
                context_value_texts = [self.gt_refs_str_list[idx] for idx in context_set_index]
            else:  # game
                context_key_texts = [self.dataset[idx]["Q"] for idx in context_set_index]
                context_value_texts = [self.dataset[idx]["description"] for idx in context_set_index]
                
            context_set_key = self.encoder.encode(context_key_texts, convert_to_tensor=True)
            context_set_val = self.encoder.encode(context_value_texts, convert_to_tensor=True)

        # 扩展到batch size
        context_set_key = context_set_key.unsqueeze(0).expand(batch_size, *context_set_key.shape)
        context_set_val = context_set_val.unsqueeze(0).expand(batch_size, *context_set_val.shape)
        
        # 组合真实KB和上下文KB
        true_kb_copy = 1
        kb_embedding = (
            torch.concat([*([train_set_key] * true_kb_copy), context_set_key], 1),
            torch.concat([*([train_set_val] * true_kb_copy), context_set_val], 1),
        )
        return kb_embedding


class Trainer:
    def __init__(
        self,
        llm_model: KBLaMPhi3ForCausalLM | KblamLlamaForCausalLM,
        kbretriever: KBRetriever,
        tokenizer: transformers.PreTrainedTokenizer,
        kb_token_layer_frequency: int,
        num_steps: int,
        lr: float,
        device: torch.device | None,
        use_lr_decay: bool,
        kb_size: int | List[int],
        llm_savename: str,
        output_dir: str,
        dataset_type: str = "law",
        sep_query_head: bool = False,
        max_seq_len: int | None = None,
    ):
        self.accelerator = Accelerator()
        self.logger = logging.getLogger("training")
        self.tokenizer = tokenizer
        self.sep_query_head = sep_query_head
        self.kb_token_layer_frequency = kb_token_layer_frequency
        self.num_steps = num_steps
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.dataset_type = dataset_type

        self.model = llm_model
        self.model.gradient_checkpointing_enable()

        self.device = device if device is not None else self.accelerator.device
        self.kbretriever = kbretriever
        self.kb_size = kb_size
        self.use_lr_decay = use_lr_decay
        self.llm_savename = llm_savename
        self.output_path = pathlib.Path(output_dir)

        # 根据模型类型选择处理函数
        if isinstance(llm_model, KBLaMPhi3ForCausalLM):  # Phi3
            self._get_batch = partial(get_batch, _format_QA_phi3, _create_labels_for_phi3, dataset_type=dataset_type)
            self._get_params = _get_phi3_query_head_parameters
        elif isinstance(llm_model, KblamLlamaForCausalLM):  # llama
            self._get_batch = partial(get_batch, _format_QA_llama, _create_labels_for_llama, dataset_type=dataset_type)
            self._get_params = _get_llama3_query_head_parameters
        else:
            raise ValueError(f"{llm_model} not recognised")

        self.scheduler, self.optim = self.setup_scheduler_and_optim()

        self.model, self.optim, self._get_batch, self.kbretriever.encoder = self.accelerator.prepare(
            self.model, self.optim, self._get_batch, self.kbretriever.encoder
        )

    def setup_scheduler_and_optim(self):
        if self.sep_query_head:
            self.logger.info("Query head being fine tuned!")
            llm_q_params = self._get_params(self.model, self.sep_query_head, self.kb_token_layer_frequency)
            scheduler, optim = setup_scheduler_and_optimizer(
                chain(self.kbretriever.encoder.parameters(), llm_q_params),
                self.lr,
                self.num_steps,
            )
            self.logger.info("Optimizer recreated")
        else:
            scheduler, optim = setup_scheduler_and_optimizer(
                self.kbretriever.encoder.parameters(), self.lr, self.num_steps
            )
            self.logger.info("Optimizer recreated")
        return scheduler, optim

    def train(
        self,
        training_set: List[Dict],
        batch_size,
        grad_accum_steps: int,
        outlier_num: int,
        use_data_aug: bool = False,
        multi_entities: bool = False,
        use_extended_qa: bool = False,
        save_period: int = 600,
        resumed_step: int = 0,
        kb_config: LKIFConfig = None,
    ):
        train_losses = []
        start_step = resumed_step

        loss_fct = CrossEntropyLoss(reduction="none")

        # 计算每个GPU的累积步骤
        num_processes = self.accelerator.num_processes
        accum_steps_per_gpu = max(1, grad_accum_steps // num_processes)
        effective_batch_size = batch_size * grad_accum_steps

        if self.accelerator.is_main_process:
            self.logger.info(f"Training with {num_processes} GPUs")
            self.logger.info(f"Total accumulation steps: {grad_accum_steps}, Steps per GPU: {accum_steps_per_gpu}")
            self.logger.info(f"Batch size: {batch_size}")
            self.logger.info(f"Effective batch size: {effective_batch_size}")

        with create_custom_progress_bar(console=console, disable=not self.accelerator.is_main_process) as pbar:
            task = pbar.add_task("Training", total=self.num_steps, loss=100)
            for step in range(start_step, self.num_steps, 1):
                self.optim.zero_grad()
                losses = []

                # 计算当前GPU需要处理的累积步骤
                process_rank = self.accelerator.process_index
                start_accum_step = process_rank * accum_steps_per_gpu
                end_accum_step = min(start_accum_step + accum_steps_per_gpu, grad_accum_steps)

                # 累积梯度
                for a_step in range(start_accum_step, end_accum_step):
                    step_config = get_step_config(
                        a_step,
                        grad_accum_steps,
                        use_data_aug,
                        outlier_num,
                        multi_entities,
                        use_extended_qa,
                    )
                    input_ids, attention_masks, labels, batch_indices = self._get_batch(
                        training_set,
                        self.tokenizer,
                        self.device,
                        B=batch_size,
                        random_sample=True,** step_config,
                    )

                    if a_step == 0 and step % 10 == 0:
                        self.logger.info(f"INPUT IDs SHAPE: {input_ids.shape}")

                    if self.max_seq_len is not None:
                        input_ids = input_ids[:, : self.max_seq_len]
                        attention_masks = attention_masks[:, : self.max_seq_len]
                        labels = labels[:, : self.max_seq_len]
                        if a_step == 0 and step % 10 == 0:
                            self.logger.info(f"TRUNCATED INPUT IDs SHAPE: {input_ids.shape}")

                    kb_embedding = self.kbretriever.get_key_embeddings(
                        batch_indices, len(input_ids), step, self.kb_size
                    )
                    out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        kb_kvs=kb_embedding,
                        output_attentions=True,
                        kb_config=kb_config,
                    )
                    logits = out["logits"]

                    # 打印部分结果用于调试
                    if a_step == 0 and step % 10 == 0:
                        batch_index = 0  # 选择batch中的第一个示例
                        max_logits = logits.argmax(axis=2)
                        decoded_pred = self.tokenizer.decode(max_logits[batch_index, :-1])
                        sel_labels = labels[batch_index, :]
                        sel_labels = sel_labels[sel_labels >= 0]  # 移除填充标记-100
                        decoded_gt = self.tokenizer.decode(sel_labels)
                        self.logger.info(f"KB SHAPE: {kb_embedding[0].shape}")
                        self.logger.info(f"GT: {decoded_gt}")
                        self.logger.info(f"PRED: {decoded_pred}")

                    # 计算损失
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    weights = (shift_labels > 0).sum(-1, keepdim=True).expand(-1, shift_labels.shape[1]).contiguous()
                    
                    model_config = (
                        self.model.config
                        if not isinstance(self.model, DistributedDataParallel)
                        else self.model.module.config
                    )
                    shift_logits = shift_logits.view(-1, model_config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    weights = weights.view(-1)

                    shift_labels = shift_labels.to(shift_logits.device)

                    loss = (
                        loss_fct(shift_logits, shift_labels) * weights.max() / weights
                    ).mean()  # 确保每个样本权重相等

                    self.accelerator.backward(loss)
                    losses.append(loss.item())

                self.optim.step()
                if self.use_lr_decay:
                    self.scheduler.step()

                # 收集并平均所有GPU的损失
                if losses:  # 仅当当前GPU处理了批次
                    local_loss = torch.tensor(np.mean(losses), device=self.device)
                else:
                    local_loss = torch.tensor(0.0, device=self.device)

                # 收集所有进程的损失
                all_losses = self.accelerator.gather(local_loss)
                valid_losses = all_losses[all_losses > 0]  # 过滤未处理批次的GPU
                avg_loss = valid_losses.mean().item() if len(valid_losses) > 0 else 0.0

                # 仅主进程记录日志
                if self.accelerator.is_main_process:
                    self.logger.info(f"step: {step}, loss: {avg_loss}")
                    train_losses.append(avg_loss)
                    pbar.update(task, advance=1, loss=avg_loss)

                # 保存检查点
                if (step % save_period) == 0 and (step != start_step):
                    try:
                        # 记录保存前的内存使用
                        self.logger.info(
                            f"Is main process: {self.accelerator.is_main_process}, GPU memory before save: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB"
                        )

                        # 尝试释放内存
                        torch.cuda.empty_cache()

                        # 同步所有进程
                        self.accelerator.wait_for_everyone()

                        if self.accelerator.is_main_process:
                            self.logger.info("Saving checkpoint...")
                            model_ckpt_name = self.output_path / f"{self.llm_savename}_step_{step}"
                            model_ckpt_name.mkdir(parents=True, exist_ok=True)

                            # 创建编码器目录
                            encoder_dir = self.output_path / f"{self.llm_savename}_step_{step}_encoder"
                            encoder_dir.mkdir(parents=True, exist_ok=True)

                            self.logger.info("Saving model...")
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            unwrapped_model.save_pretrained(
                                model_ckpt_name,
                                is_main_process=self.accelerator.is_main_process,
                                save_function=self.accelerator.save,
                            )

                            self.logger.info("Saving encoder...")
                            encoder_ckpt_name = encoder_dir / "encoder.pt"
                            torch.save(self.kbretriever.encoder.state_dict(), encoder_ckpt_name)

                            self.logger.info("Saving config...")
                            config_path = model_ckpt_name / "kb_config_explicit.json"
                            with open(config_path, 'w') as f:
                                f.write(kb_config.to_json_string())

                    except Exception as e:
                        self.logger.error(f"Error saving checkpoint: {e}")
                        self.logger.error(f"Error details: {str(e)}")
                        raise e


def main():
    os.environ["NCCL_TIMEOUT"] = "1200000"
    logger = logging.getLogger("training")

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    print(vars(args))
    dataset_name = args.train_dataset
    seed = args.seed
    N = args.N
    B = args.B

    total_steps = args.total_steps
    encoder_spec = args.encoder_spec
    key_embd_src = args.key_embd_src
    use_data_aug = args.use_data_aug
    use_lr_decay = args.use_lr_decay
    use_cached_embd = args.use_cached_embd
    dataset_dir = args.dataset_dir
    model_dir_to_resume = args.model_dir_to_resume
    model_save_dir = args.model_save_dir
    sep_query_head = args.sep_query_head
    kb_size = args.kb_size
    dynamic_kb_size = args.dynamic_kb_size
    max_seq_len = args.max_seq_len
    dataset_type = args.dataset_type

    if kb_size is not None and dynamic_kb_size is not None:
        raise ValueError("Can't specify kb_size and dynamic_kb_size. Use only one")

    kb_size = kb_size if kb_size is not None else dynamic_kb_size

    gradient_accm_step = args.gradient_accm_step

    length_invariance = args.length_invariance
    outlier_num = args.outlier_num
    multi_entities = args.multi_entities
    use_extended_qa = args.use_extended_qa
    kb_token_layer_frequency = args.kb_token_layer_frequency
    llm_type = args.llm_type
    hf_model_spec = args.hf_model_spec
    hf_token = args.hf_token
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 创建保存目录
    pathlib.Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    if Accelerator().is_main_process:
        pass  # 禁用wandb

    # 尝试释放内存
    torch.cuda.empty_cache()

    if args.log_to_file:
        formatter = logging.Formatter(LOGFORMAT)
        f_handler = logging.FileHandler(os.path.join(model_save_dir, "log.txt"))
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    logger.info(f"Running on {device}")
    logger.info("Started training")
    logger.info(f"Saving to  {model_save_dir}")
    
    if sep_query_head:
        os.environ["SEP_QUERY_HEAD"] = "TRUE"
        logger.info("Having separate query head for KB!")

    if length_invariance:
        os.environ["LENGTH_INVARIANCE"] = "TRUE"
        logger.info("Using length invariance scaling!")

    os.environ["SCALE_FACTOR"] = ""

    # 加载缓存的嵌入
    key_embds = None
    value_embds = None
    if use_cached_embd:
        logger.info(f"Using pre-computed {encoder_spec} embedding")
        key_embds, value_embds = _load_cached_embeddings(encoder_spec, dataset_dir, dataset_name, key_embd_src)

    # 获取实验前缀
    prefix_string = get_prefix_str(args)
    logger.info(f"Experiment prefix {prefix_string}")

    # 加载并处理数据集
    if use_extended_qa:
        dataset_path = os.path.join(dataset_dir, f"{dataset_name}_augmented.json")
    else:
        dataset_path = os.path.join(dataset_dir, f"{dataset_name}.json")
    
    # 支持JSON和JSONL格式
    dataset = []
    with open(dataset_path, 'r') as f:
        content = f.read().strip()
        try:
            # 尝试JSON格式
            json_data = json.loads(content)
            if isinstance(json_data, list):
                dataset = json_data
            else:
                dataset = [json_data]
        except json.JSONDecodeError:
            # 尝试JSONL格式
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        dataset.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line: {e}")
                        continue
    
    # 根据数据集类型处理字段
    processed_dataset = []
    for data in dataset:
        if dataset_type == "law":
            # 法律类数据集处理：仅保留实际存在且训练需要的字段
            processed_data = {
                "query": data.get("query", {}),
                "ground_truth": data.get("ground_truth", {}),
            }
        else:
            # 游戏类数据集处理
            processed_data = data

        processed_dataset.append(processed_data)
    
    training_set = processed_dataset[:N]
    logger.info(f"Loaded {len(training_set)} samples from {dataset_path}")

    # 设置LLM
    llm_model_spec = model_dir_to_resume if model_dir_to_resume else hf_model_spec
    resumed_step = 0 if not model_dir_to_resume else int(model_dir_to_resume.split("_")[-1])

    if llm_model_spec is None:
        raise ValueError("Either supply model_dir_to_resume or hf_model_spec")

    # 检查是否使用HuggingFace模型
    is_hf_model = not os.path.exists(hf_model_spec) and "/" in hf_model_spec and not hf_model_spec.startswith("/")
    
    if hf_token is None and args.llm_type == "llama3" and is_hf_model:
        raise ValueError("Please supply HuggingFace token(hf_token) when loading Llama weights from HuggingFace")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_spec,
        trust_remote_code=True,
        token=hf_token if hf_token and args.llm_type == "llama3" and is_hf_model else None,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    if args.llm_type == "llama3":
        model = KblamLlamaForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    elif args.llm_type == "phi3":
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            llm_model_spec,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"LLM type {args.llm_type} not recognised")

    logger.info(model.config)  # type: ignore

    # 冻结模型参数
    model.eval()  # type: ignore
    for _, param in model.named_parameters():  # type: ignore
        param.requires_grad = False

    # 设置编码器
    encoder = LKIFEncoder(
        encoder_name=encoder_spec,
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size *  # type: ignore
        (model.config.num_hidden_layers // kb_token_layer_frequency + 1),  # type: ignore
        frozen_base_model=True,
        device=device,
    )

    # 加载 resume 模型
    if model_dir_to_resume:
        encoder.load_state_dict(torch.load(os.path.join(model_dir_to_resume, "encoder.pt")))
        kb_config = LKIFConfig.from_pretrained(os.path.join(model_dir_to_resume, "kb_config.json"))
    else:
        kb_config = LKIFConfig(
            sep_query_head=sep_query_head,
            kb_layer_frequency=kb_token_layer_frequency,
        )

    encoder.train()

    # 创建KB检索器
    kbretriever = KBRetriever(
        encoder,
        training_set,
        key_embds=key_embds,  # type: ignore
        value_embds=value_embds,  # type: ignore
        dataset_type=dataset_type
    )

    logger.info("Model ready")

    # 开始训练
    llm_ckpt_name = f"{prefix_string}KeyFrom{key_embd_src}_{encoder_spec}_{dataset_name}_{llm_type}"

    trainer = Trainer(
        model,  # type: ignore
        kbretriever,
        tokenizer,
        kb_token_layer_frequency,
        total_steps,
        args.lr,
        device,
        use_lr_decay,
        kb_size,  # type: ignore
        llm_ckpt_name,
        model_save_dir,
        dataset_type=dataset_type,
        sep_query_head=sep_query_head,
        max_seq_len=max_seq_len,
    )

    logger.info(f"Number of trainable parameters: {_get_parameter_count(encoder):,}")

    trainer.train(
        training_set,
        B,
        gradient_accm_step,
        outlier_num,
        use_data_aug=use_data_aug,
        multi_entities=multi_entities,
        use_extended_qa=use_extended_qa,
        save_period=600,
        resumed_step=resumed_step,
        kb_config=kb_config,
    )


if __name__ == "__main__":
    main()
