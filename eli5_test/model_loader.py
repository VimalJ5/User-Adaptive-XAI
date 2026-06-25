"""
model_loader.py
===============
CUDA and 4-bit loading utilities for the Qwen2.5 generation model.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cd_generator import ReadabilityBeamGenerator
from experiment_config import CONFIG
from utils import format_bytes


def print_vram_status() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Experiment A, but torch.cuda is not available.")
    device_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"[VRAM] Device: {device_name}")
    print(f"[VRAM] Free: {format_bytes(int(free_bytes))} | Total: {format_bytes(int(total_bytes))}")


def load_qwen_model():
    print_vram_status()
    print(f"[Loader] Loading Qwen model from '{CONFIG['model_name']}' in 4-bit ...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(CONFIG["load_in_4bit"]),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model.eval()

    footprint_bytes = int(model.get_memory_footprint())
    print(f"[Loader] Model memory footprint: {format_bytes(footprint_bytes)}")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"[VRAM] After load free: {format_bytes(int(free_bytes))} | Total: {format_bytes(int(total_bytes))}")
    return tokenizer, model


def build_cd_generator(model, tokenizer) -> ReadabilityBeamGenerator:
    return ReadabilityBeamGenerator(model=model, tokenizer=tokenizer, num_beams=CONFIG["num_beams"])
