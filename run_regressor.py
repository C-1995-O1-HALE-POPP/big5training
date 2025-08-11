# -*- coding: utf-8 -*-
"""
批量评估：在保留原始 JSON 结构与字段的前提下，为每个含 response 的叶子元素追加
    regressor_score / regressor_label
"""

import os
import json
import argparse
from typing import Any, Dict, List
from tqdm import tqdm
from loguru import logger
import sys

# !!! 修改为你 big5_regressor 的实际导入位置 !!!
from big5regressor.inference import big5_regressor

# ===== 日志配置 =====
logger.remove()
logger.add(
    sink=sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>"
)
logger.add(
    sink="logs/run_regressor_preserve_full.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)

TRAITS = ["O", "C", "E", "A", "N"]


def _augment_leaf_list(
    clf: "big5_regressor",
    leaf: List[Dict[str, Any]],
    trait: str,
    threshold: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """
    对含 response 的叶子列表做批量推理，在每个元素上原地追加：
        regressor_score, regressor_label
    不删除任何原字段；保持原顺序。
    """
    texts, idx_map = [], []
    for i, item in enumerate(leaf):
        if isinstance(item, dict) and "response" in item:
            texts.append(item["response"])
            idx_map.append(i)

    if not texts:
        return leaf  # 原样返回

    # 批量推理
    all_scores: List[Dict[str, Dict[str, Any]]] = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s : s + batch_size]
        batch_scores = clf.inference(batch)  # [{'O':{'score','label'}, ...}, ...]
        all_scores.extend(batch_scores)

    # 追加字段
    for rel_idx, i in enumerate(idx_map):
        per_text_scores = all_scores[rel_idx]
        t_score = float(per_text_scores[trait]["score"])
        t_label = per_text_scores[trait].get("label", "high" if t_score >= threshold else "low")
        # 不覆盖已有字段名，采用独立命名
        leaf[i]["regressor_score"] = t_score
        leaf[i]["regressor_label"] = t_label

    return leaf


def _preserve_transform_group(
    group: Any,
    trait: str,
    clf: "big5_regressor",
    threshold: float,
    batch_size: int
) -> Any:
    """
    递归遍历，完整保留结构与字段：
    - dict: 逐键递归
    - list: 若元素含 response，则批量推理并在元素上追加 regressor_* 字段；否则保持原样
    - 其他类型: 原样返回
    """
    if isinstance(group, list):
        has_resp = any(isinstance(x, dict) and "response" in x for x in group)
        return _augment_leaf_list(clf, group, trait, threshold, batch_size) if has_resp else group

    if isinstance(group, dict):
        return {k: _preserve_transform_group(v, trait, clf, threshold, batch_size) for k, v in group.items()}

    return group


def evaluate_and_preserve_all_fields(
    obj: Dict[str, Any],
    clf: "big5_regressor",
    threshold: float,
    batch_size: int
) -> Dict[str, Any]:
    """
    顶层应含 O/C/E/A/N。保留所有层级与字段，仅给叶子元素追加 regressor_*。
    """
    if not (isinstance(obj, dict) and all(k in obj for k in TRAITS)):
        logger.warning("输入 JSON 顶层不符合预期（缺少 O/C/E/A/N），将原样返回。")
        return obj

    out: Dict[str, Any] = {}
    for trait in TRAITS:
        group = obj.get(trait, {})
        if isinstance(group, (dict, list)):
            out[trait] = _preserve_transform_group(group, trait, clf, threshold, batch_size)
        else:
            out[trait] = group
    return out


# ========= per-file wrappers (保持和你之前分类一致) =========

def process_evaluate_score_file(in_path: str, out_path: str, clf, threshold: float, batch_size: int):
    logger.info(f"[evaluate_score] 读取: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    res = evaluate_and_preserve_all_fields(obj, clf, threshold, batch_size)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    logger.success(f"[evaluate_score] 写出: {out_path}")


def process_single_question_lora_test_result_file(in_path: str, out_path: str, clf, threshold: float, batch_size: int):
    logger.info(f"[single_question_lora_test_result] 读取: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    res = evaluate_and_preserve_all_fields(obj, clf, threshold, batch_size)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    logger.success(f"[single_question_lora_test_result] 写出: {out_path}")


def process_single_question_merge_lora_test_result_file(in_path: str, out_path: str, clf, threshold: float, batch_size: int):
    logger.info(f"[single_question_merge_lora_test_result] 读取: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    res = evaluate_and_preserve_all_fields(obj, clf, threshold, batch_size)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    logger.success(f"[single_question_merge_lora_test_result] 写出: {out_path}")


def process_single_question_lora_prompt_file(in_path: str, out_path: str, clf, threshold: float, batch_size: int):
    logger.info(f"[single_question_lora_prompt] 读取: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    res = evaluate_and_preserve_all_fields(obj, clf, threshold, batch_size)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    logger.success(f"[single_question_lora_prompt] 写出: {out_path}")


def process_single_question_mergelora_prompt_file(in_path: str, out_path: str, clf, threshold: float, batch_size: int):
    logger.info(f"[single_question_mergelora_prompt] 读取: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    res = evaluate_and_preserve_all_fields(obj, clf, threshold, batch_size)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    logger.success(f"[single_question_mergelora_prompt] 写出: {out_path}")


def process_generic_file(in_path: str, out_path: str, clf, threshold: float, batch_size: int):
    logger.info(f"[generic] 读取: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    res = evaluate_and_preserve_all_fields(obj, clf, threshold, batch_size)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    logger.success(f"[generic] 写出: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="big5_regressor.pt 路径")
    ap.add_argument("--input_dir", required=True, help="输入 JSON 目录")
    ap.add_argument("--output_dir", required=True, help="输出目录（保留结构，叶子元素追加 regressor_* 字段）")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"加载回归器模型: {args.model_path}")
    clf = big5_regressor(
        model_path=args.model_path,
        base_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_length=args.max_length,
        batch_size=args.batch_size,
        threshold=args.threshold,
        chunk_long_text=False,
        device=args.device
    )

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    logger.info(f"共发现 {len(files)} 个 JSON 文件待处理")

    for fn in tqdm(files, desc="处理 JSON 文件", unit="file"):
        in_path = os.path.join(args.input_dir, fn)
        name = os.path.splitext(fn)[0]
        out_path = os.path.join(args.output_dir, f"{name}_with_regressor_preserved.json")

        lower = name.lower()
        if "evaluate_score" in lower:
            process_evaluate_score_file(in_path, out_path, clf, args.threshold, args.batch_size)
        elif "single_question_merge_lora_test_result" in lower:
            process_single_question_merge_lora_test_result_file(in_path, out_path, clf, args.threshold, args.batch_size)
        elif "single_question_lora_test_result" in lower:
            process_single_question_lora_test_result_file(in_path, out_path, clf, args.threshold, args.batch_size)
        elif "single_question_mergelora_prompt" in lower:
            process_single_question_mergelora_prompt_file(in_path, out_path, clf, args.threshold, args.batch_size)
        elif "single_question_lora_prompt" in lower:
            process_single_question_lora_prompt_file(in_path, out_path, clf, args.threshold, args.batch_size)
        else:
            process_generic_file(in_path, out_path, clf, args.threshold, args.batch_size)


if __name__ == "__main__":
    main()
