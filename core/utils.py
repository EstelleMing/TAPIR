"""种子与 YAML 配置覆盖。命令行显式传入的参数优先于配置文件。"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _flatten_yaml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把分节 YAML 展平。同名键以后出现的节为准。"""
    flat: Dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def explicit_destinations(parser: argparse.ArgumentParser) -> set:
    """sys.argv 里真正写过的参数名。"""
    seen = set()
    argv = set(sys.argv[1:])
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in argv:
                seen.add(action.dest)
    return seen


def overlay_yaml(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """若提供 --config_file，用 YAML 填充未被命令行显式设置的字段。"""
    path = getattr(args, "config_file", "") or ""
    if not path:
        return args
    import yaml

    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    flat = _flatten_yaml(cfg)
    locked = explicit_destinations(parser)
    for key, value in flat.items():
        if key in ("config_file",) or key in locked:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    return args
