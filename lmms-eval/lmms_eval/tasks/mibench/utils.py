# import os
# import re
# import logging
# from PIL import Image

# eval_logger = logging.getLogger("lmms-eval")

# def _root():
#     return os.environ.get("MIBENCH_ROOT", "")

# def _resolve(p: str) -> str:
#     r = _root()
#     if r and not os.path.isabs(p):
#         return os.path.join(r, p)
#     return p

# def mii_doc_to_text(doc, lmms_eval_specific_kwargs=None):
#     lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
#     pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
#     post = lmms_eval_specific_kwargs.get("post_prompt", "")

#     q = str(doc["question"])
#     options = doc.get("options", [])
#     if options is None:
#         options = []
#     options = [str(o).strip() for o in options]

#     # 自检：<image> 个数 vs 图片数
#     n_ph = q.count("<image>")
#     n_img = len(doc.get("image", []) or [])
#     if n_ph != n_img and n_img > 0:
#         eval_logger.warning(f"[MII] placeholder mismatch id={doc.get('id')}: <image>={n_ph}, images={n_img}")

#     opt_text = "\nOptions:\n" + "\n".join([f"- {o}" for o in options]) if options else ""
#     return f"{pre}{q}{opt_text}{post}"

# def mii_doc_to_visual(doc):
#     paths = doc.get("image", []) or []
#     imgs = []
#     for p in paths:
#         imgs.append(Image.open(_resolve(p)).convert("RGB"))
#     return imgs

# def mii_doc_to_target(doc):
#     # 这里保持字符串即可（你的 multi_image_instruction 应该是 MCQ）
#     return str(doc["answer"]).strip()

# def _parse_pred_to_option(pred: str, options: list[str]) -> str:
#     if not pred or not options:
#         return ""
    
#     raw = pred.strip()
#     raw_l = raw.lower()

#     # 1. 严格匹配：模型输出和某个选项完全一致（忽略大小写和首尾空格）
#     for opt in options:
#         if raw_l == opt.strip().lower():
#             return opt

#     # 2. 关键词匹配：检查选项文本是否出现在模型的回答中
#     # 按照选项长度倒序排列，防止短选项被错误匹配（例如 "cat" 和 "black cat"）
#     sorted_options = sorted(options, key=len, reverse=True)
#     for opt in sorted_options:
#         opt_l = opt.strip().lower()
#         # 使用正则表达式匹配完整的选项短语，防止单词内匹配
#         if re.search(rf"\b{re.escape(opt_l)}\b", raw_l):
#             return opt

#     # 3. 如果模型依然输出了 A/B/C/D（尽管你没给，但有些模型有这种惯性）
#     # 这一步作为兜底，如果模型自作聪明加了编号也能救回来
#     m = re.search(r"\b([A-Z])\b", raw)
#     if m:
#         idx = ord(m.group(1)) - ord("A")
#         if 0 <= idx < len(options):
#             return options[idx]

#     return ""

# def mii_process_results(doc, results):
#     pred = results[0]
#     options = doc.get("options", []) or []
#     options = [str(o).strip() for o in options]
#     gold = str(doc["answer"]).strip()

#     parsed = _parse_pred_to_option(pred, options)
#     correct = (parsed.strip().lower() == gold.strip().lower())

#     return {
#         "acc": {
#             "id": doc.get("id", ""),
#             "task": str(doc.get("task", "unknown")),
#             "pred_raw": pred,
#             "pred_parsed": parsed,
#             "gold": gold,
#             "correct": correct,
#         }
#     }


# def mii_aggregation(results):
#     """
#     results: list[dict]，每个 dict 是 process_results 里 acc 的 value
#     """
#     if not results:
#         return 0.0

#     total = len(results)
#     total_correct = sum(1 for r in results if r.get("correct", False))
#     overall = total_correct / total

#     # per-task stats
#     by_task = {}
#     for r in results:
#         t = r.get("task", "unknown") or "unknown"
#         by_task.setdefault(t, {"n": 0, "c": 0})
#         by_task[t]["n"] += 1
#         by_task[t]["c"] += 1 if r.get("correct", False) else 0

#     # 打印：每个 task 的 acc
#     print("Performances by task:")
#     print("=" * 60)
#     for t in sorted(by_task.keys()):
#         n = by_task[t]["n"]
#         c = by_task[t]["c"]
#         print(f"{t:<35} : {c/n:.4f}  ({c}/{n})")
#     print("=" * 60)
#     print(f"{'OVERALL':<35} : {overall:.4f}  ({total_correct}/{total})")
#     print("=" * 60)

#     return overall




import os
import re
import logging
import string
from PIL import Image

eval_logger = logging.getLogger("lmms-eval")


def _root():
    return os.environ.get("MIBENCH_ROOT", "")


def _resolve(p: str) -> str:
    r = _root()
    if r and not os.path.isabs(p):
        return os.path.join(r, p)
    return p


def _get_image_paths(doc):
    paths = doc.get("image", None)
    if paths is None:
        paths = doc.get("images", None)

    if paths is None:
        return []

    if isinstance(paths, str):
        return [paths]

    return list(paths)


def _normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    text = text.strip(string.punctuation + " ")
    return text


def _clean_option_text(opt: str) -> str:
    return str(opt).strip()


def mii_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post = lmms_eval_specific_kwargs.get("post_prompt", "")

    q = str(doc["question"]).strip()
    options = doc.get("options", []) or []
    options = [_clean_option_text(o) for o in options]

    n_ph = q.count("<image>")
    n_img = len(_get_image_paths(doc))
    if n_img > 0 and n_ph != n_img:
        eval_logger.warning(
            f"[MIBENCH] placeholder mismatch id={doc.get('id')}: <image>={n_ph}, images={n_img}"
        )

    opt_text = ""
    if options:
        opt_text = "\n\nOptions:\n" + "\n".join(options)

    return f"{pre}{q}{opt_text}{post}"


def mii_doc_to_visual(doc):
    paths = _get_image_paths(doc)
    imgs = []
    for p in paths:
        imgs.append(Image.open(_resolve(p)).convert("RGB"))
    return imgs


def mii_doc_to_target(doc):
    return str(doc["answer"]).strip()


def _parse_pred_to_option(pred: str, options: list[str]) -> str:
    if not pred or not options:
        return ""

    raw = str(pred).strip()
    raw_norm = _normalize_text(raw)

    cleaned_options = [_clean_option_text(o) for o in options]
    option_norm_map = {opt: _normalize_text(opt) for opt in cleaned_options}

    # 1. 输出与某个选项完全一致
    for opt in cleaned_options:
        if raw_norm == option_norm_map[opt]:
            return opt

    # 2. 常见前缀形式
    patterns = [
        r"^answer\s*(?:is|:)?\s*(.+)$",
        r"^the answer is\s+(.+)$",
        r"^option\s*(?:is|:)?\s*(.+)$",
        r"^choice\s*(?:is|:)?\s*(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, raw_norm)
        if m:
            tail = _normalize_text(m.group(1))
            for opt in cleaned_options:
                if tail == option_norm_map[opt]:
                    return opt

    # 3. 回答中包含某个完整选项短语
    sorted_options = sorted(cleaned_options, key=lambda x: len(option_norm_map[x]), reverse=True)
    for opt in sorted_options:
        opt_norm = option_norm_map[opt]
        if opt_norm and re.search(rf"\b{re.escape(opt_norm)}\b", raw_norm):
            return opt

    # 4. 兜底：模型若输出 A/B/C/D，则映射到第几个 option
    m = re.search(r"\b([A-Z])\b", raw.strip())
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(cleaned_options):
            return cleaned_options[idx]

    return ""


def mii_process_results(doc, results):
    pred = results[0] if isinstance(results, list) else results
    options = doc.get("options", []) or []
    options = [_clean_option_text(o) for o in options]
    gold = str(doc["answer"]).strip()

    parsed = _parse_pred_to_option(pred, options)
    correct = (_normalize_text(parsed) == _normalize_text(gold))

    return {
        "acc": {
            "id": doc.get("id", ""),
            "task": str(doc.get("task", "unknown")),
            "pred_raw": pred,
            "pred_parsed": parsed,
            "gold": gold,
            "correct": correct,
        }
    }


def mii_aggregation(results):
    if not results:
        return 0.0

    total = len(results)
    total_correct = sum(1 for r in results if r.get("correct", False))
    overall = total_correct / total

    by_task = {}
    for r in results:
        t = r.get("task", "unknown") or "unknown"
        by_task.setdefault(t, {"n": 0, "c": 0})
        by_task[t]["n"] += 1
        by_task[t]["c"] += 1 if r.get("correct", False) else 0

    print("Performances by task:")
    print("=" * 60)
    for t in sorted(by_task.keys()):
        n = by_task[t]["n"]
        c = by_task[t]["c"]
        print(f"{t:<35} : {c / n:.4f}  ({c}/{n})")
    print("=" * 60)
    print(f"{'OVERALL':<35} : {overall:.4f}  ({total_correct}/{total})")
    print("=" * 60)

    return overall