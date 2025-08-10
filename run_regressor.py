# -*- coding: utf-8 -*-
"""
批量评估：仅保留“原文 + 对应维度的 regressor 评分”
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple
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
    sink="logs/run_regressor_min.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)

TRAITS = ["O", "C", "E", "A", "N"]

def iter_texts_with_trait(obj: Any) -> List[Tuple[str, str]]:
    """返回 [(text, trait), ...]"""
    pairs: List[Tuple[str, str]] = []
    if not (isinstance(obj, dict) and all(k in obj for k in TRAITS)):
        return pairs

    for trait, group in obj.items():
        if not isinstance(group, dict):
            continue
        for _, v in group.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "response" in item:
                        pairs.append((item["response"], trait))
            elif isinstance(v, dict):
                for lst in v.values():
                    if isinstance(lst, list):
                        for item in lst:
                            if isinstance(item, dict) and "response" in item:
                                pairs.append((item["response"], trait))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="big5_regressor.pt 路径")
    ap.add_argument("--input_dir", required=True, help="输入 JSON 目录")
    ap.add_argument("--output_dir", required=True, help="输出目录（仅 text + 对应维度评分）")
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
        logger.info(f"读取文件: {in_path}")
        with open(in_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        pairs = iter_texts_with_trait(obj)
        if not pairs:
            logger.warning(f"{fn} 未找到可评估的 response，跳过")
            continue

        texts = [t for t, _ in pairs]
        logger.info(f"{fn} - 待推理文本数: {len(texts)}")

        # 推理带进度条
        all_scores = []
        for i in tqdm(range(0, len(texts), args.batch_size), desc="推理中", leave=False):
            batch = texts[i:i + args.batch_size]
            batch_scores = clf.inference(batch)
            all_scores.extend(batch_scores)

        minimal_out: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TRAITS}
        for (text, trait), per_text_scores in zip(pairs, all_scores):
            trait_score = per_text_scores[trait]
            minimal_out[trait].append({
                "text": text,
                "score": float(trait_score["score"]),
                "label": trait_score.get("label", "high" if trait_score["score"] >= args.threshold else "low")
            })

        out_path = os.path.join(args.output_dir, fn.replace(".json", "_with_regressor_min.json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(minimal_out, f, ensure_ascii=False, indent=2)

        logger.success(f"写出: {out_path}")

if __name__ == "__main__":
    main()
