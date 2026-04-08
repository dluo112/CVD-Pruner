# import re

# def doc_to_visual(doc):
#     # HF datasets 已经解码成 list[PIL.Image]，直接返回即可
#     return doc["images"]

# def _extract_letters_from_options(options):
#     # options 形如 "(A) ...", "(B) ..."
#     letters = []
#     for o in options:
#         m = re.match(r"\s*\(?([A-Z])\)?\s*[\)\.:]", o.strip())
#         if m:
#             letters.append(m.group(1))
#     # fallback: A,B,C...
#     if not letters:
#         letters = [chr(ord("A") + i) for i in range(len(options))]
#     return letters

# def doc_to_text(doc):
#     q = doc["question"].strip()
#     qtype = doc["question_type"]

#     if qtype == "multi-choice":
#         opts = doc["options"]
#         letters = _extract_letters_from_options(opts)

#         q += "\n\nOptions:\n"
#         for i, o in enumerate(opts):
#             # 统一展示成 "A. xxx"
#             q += f"{letters[i]}. {o}\n"

#         # 强制输出单个字母，保证 exact_match 稳
#         q += f"\nAnswer with a single letter only: {', '.join(letters)}."
#     else:
#         # 如果后续有别的类型（比如 free-form），先让它简答
#         q += "\n\nAnswer briefly."
#     return q

# def doc_to_target(doc):
#     # 你的例子 answer 就是 "C"
#     return str(doc["answer"]).strip()
# import re
# import string


# def doc_to_visual(doc):
#     """
#     HF datasets 通常会把 images 解码成 list[PIL.Image]。
#     lmms-eval 多图任务直接返回这个 list 即可。
#     """
#     return doc["images"]


# def _normalize_ws(text):
#     return " ".join(str(text).strip().split())


# def _normalize_text(text):
#     """
#     给 short-answer 用的宽松归一化：
#     - 小写
#     - 去前后空白
#     - 去多余标点
#     - 压缩空格
#     """
#     text = str(text).strip().lower()
#     text = text.replace("\n", " ")
#     text = _normalize_ws(text)

#     # 去掉首尾标点
#     text = text.strip(string.punctuation + " ")
#     return text


# def _strip_option_prefix(option_text):
#     """
#     把 '(A) cat' / 'A. cat' / 'B: dog' 这种前缀去掉，
#     最终只保留 'cat' / 'dog'
#     """
#     text = str(option_text).strip()
#     text = re.sub(r"^\s*\(?([A-Z])\)?\s*[\)\].:\-]?\s*", "", text)
#     return text.strip()


# def _extract_letters_from_options(options):
#     """
#     从 options 中提取选项字母。
#     例如:
#       '(A) cat' -> A
#       'B. dog'  -> B
#     如果提取不到，就 fallback 成 A/B/C/...
#     """
#     letters = []
#     for i, o in enumerate(options):
#         m = re.match(r"^\s*\(?([A-Z])\)?\s*[\)\].:\-]?", str(o).strip())
#         if m:
#             letters.append(m.group(1))
#         else:
#             letters.append(chr(ord("A") + i))
#     return letters


# def _build_option_map(options):
#     """
#     返回:
#       letters: ['A','B',...]
#       option_map: {
#          'A': 'cat',
#          'B': 'dog',
#       }
#     """
#     letters = _extract_letters_from_options(options)
#     option_map = {}
#     for letter, option in zip(letters, options):
#         option_map[letter] = _strip_option_prefix(option)
#     return letters, option_map


# def _canonicalize_mc_answer(answer, options):
#     """
#     把 gold answer 规范成单个字母:
#     - 如果本身就是 A/B/C/D，直接返回
#     - 如果是完整选项文本，尝试映射回对应字母
#     """
#     letters, option_map = _build_option_map(options)

#     ans = str(answer).strip()

#     # 情况1：answer 本身就是 A / B / C / D
#     m = re.match(r"^\s*\(?([A-Z])\)?\s*$", ans, flags=re.IGNORECASE)
#     if m:
#         pred_letter = m.group(1).upper()
#         if pred_letter in letters:
#             return pred_letter

#     # 情况2：answer 可能是 "(C)" / "C." / "C:"
#     m = re.match(r"^\s*\(?([A-Z])\)?\s*[\)\].:\-]?\s*$", ans, flags=re.IGNORECASE)
#     if m:
#         pred_letter = m.group(1).upper()
#         if pred_letter in letters:
#             return pred_letter

#     # 情况3：answer 是完整文本
#     norm_ans = _normalize_text(_strip_option_prefix(ans))
#     for letter, opt_text in option_map.items():
#         if _normalize_text(opt_text) == norm_ans:
#             return letter

#     # 实在匹配不上就原样大写返回，后面 process_results 再兜底
#     return ans.upper()


# def _extract_mc_prediction(pred, options):
#     """
#     从模型输出中尽量抽出单个选项字母。
#     兼容以下形式：
#       'A'
#       '(A)'
#       'Answer: B'
#       'The answer is C.'
#       'cat'  -> 如果和某个 option 文本匹配，也映射成对应字母
#     """
#     letters, option_map = _build_option_map(options)
#     text = str(pred).strip()

#     # 1) 优先找独立字母
#     patterns = [
#         r"^\s*\(?([A-Z])\)?\s*$",
#         r"answer\s*(?:is|:)?\s*\(?([A-Z])\)?",
#         r"option\s*\(?([A-Z])\)?",
#         r"choice\s*\(?([A-Z])\)?",
#         r"\b([A-Z])\b",
#     ]
#     for pat in patterns:
#         m = re.search(pat, text, flags=re.IGNORECASE)
#         if m:
#             cand = m.group(1).upper()
#             if cand in letters:
#                 return cand

#     # 2) 如果模型输出的是完整选项文本，映射回字母
#     norm_text = _normalize_text(_strip_option_prefix(text))
#     for letter, opt_text in option_map.items():
#         if _normalize_text(opt_text) == norm_text:
#             return letter

#     # 3) 有时模型会输出更长句子，检查是否包含某个 option 文本
#     for letter, opt_text in option_map.items():
#         norm_opt = _normalize_text(opt_text)
#         if norm_opt and norm_opt in norm_text:
#             return letter

#     return text.strip().upper()


# def doc_to_text(doc):
#     """
#     输入字段结构：
#       id, question_type, question, images, options, answer, data_source, category
#     """
#     q = _normalize_ws(doc["question"])
#     qtype = str(doc["question_type"]).strip().lower()

#     if qtype == "multi-choice":
#         options = doc["options"]
#         letters, option_map = _build_option_map(options)

#         prompt = q + "\n\nOptions:\n"
#         for letter in letters:
#             prompt += f"{letter}. {option_map[letter]}\n"

#         prompt += "\nAnswer with a single letter only."
#         return prompt

#     elif qtype == "short-answer":
#         prompt = q + "\n\nAnswer briefly with a single word or short phrase."
#         return prompt

#     else:
#         # 兜底
#         prompt = q + "\n\nAnswer briefly."
#         return prompt


# def doc_to_target(doc):
#     qtype = str(doc["question_type"]).strip().lower()

#     if qtype == "multi-choice":
#         return _canonicalize_mc_answer(doc["answer"], doc["options"])
#     else:
#         return _normalize_text(doc["answer"])


# def process_results(doc, results):
#     """
#     results 一般是 model.generate 的输出列表，取第一个即可。
#     返回 exact_match: 1/0
#     """
#     pred = results[0] if isinstance(results, list) else results
#     qtype = str(doc["question_type"]).strip().lower()

#     if qtype == "multi-choice":
#         gold = _canonicalize_mc_answer(doc["answer"], doc["options"])
#         pred = _extract_mc_prediction(pred, doc["options"])
#         score = 1.0 if pred == gold else 0.0
#         return {"exact_match": score}

#     else:
#         gold = _normalize_text(doc["answer"])
#         pred = _normalize_text(pred)
#         score = 1.0 if pred == gold else 0.0
#         return {"exact_match": score}

import logging
import re
from typing import List

eval_logger = logging.getLogger("lmms-eval")


def mantis_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_type, question, option = (
        doc["question_type"],
        doc["question"],
        doc["options"],
    )
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if question_type == "short-answer":
        option = ""
        final_question = f'Given the images, answer the following short answer vqa question:\nQ: {question}\nYou can first give your analysis, and then give your final answer as "Final Answer:"'
    if question_type == "multi-choice":
        final_question = f"{question}\nAnswer with the option's letter from the given choices directly."

    return f"{pre_prompt}{final_question}{option}{post_prompt}"


def mantis_doc_to_visual(doc):
    image_list = [image.convert("RGB") for image in doc["images"]]
    return image_list


def mantis_doc_to_target(doc):
    return doc["answer"]


def parse_multi_choice_response(response):
    option_letter_regex = re.compile(r"^\s*([A-Z])\.")
    match = option_letter_regex.match(response)
    if match:
        filtered_resps = match.group(1)
    else:
        filtered_resps = response
    return filtered_resps


def parse_answer(raw_answer):
    if "final answer:" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("final answer:") + len("final answer:") :].strip()
    elif "the answer is" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("the answer is") + len("the answer is") :].strip()
    elif "answer:" in raw_answer.lower():
        answer = raw_answer[raw_answer.lower().index("answer:") + len("answer:") :].strip()
    else:
        answer = raw_answer
    return answer


def get_option(final_answer):
    if re.match(r"Answer: [A-Z]", final_answer):
        return final_answer[8]
    for s in final_answer:
        if s.isalpha():
            return s.upper()
    return None


def get_prediction(question_type: str, raw_answer: str, ref_answer: str, options: List[str]):
    answer = parse_answer(raw_answer)
    ref_answer = ref_answer.strip("()\n ")  # important for some datasets
    if question_type == "multi-choice":
        if not len(ref_answer) == 1:
            for c in ref_answer:
                if c.isalpha():
                    ref_answer = c
                    break
        assert len(ref_answer) == 1, f"Ref answer is not a single character: {ref_answer}"

        selected_option = get_option(answer)
        if selected_option and (ord(selected_option) - ord("A") < len(options)):
            correct = selected_option == ref_answer.upper()
            parsed_answer = selected_option
        else:
            ref_option_idx = ord(ref_answer.upper()) - ord("A")
            if ref_option_idx >= len(options):
                correct = False
                parsed_answer = raw_answer
            else:
                ref_raw_answer = options[ref_option_idx]
                if ref_raw_answer.startswith(ref_answer + "."):
                    correct = raw_answer.strip() == ref_raw_answer[len(ref_answer + ".") :].strip()
                elif ref_raw_answer.startswith(ref_answer + ":"):
                    correct = raw_answer.strip() == ref_raw_answer[len(ref_answer + ":") :].strip()
                elif ref_raw_answer.startswith("(" + ref_answer + ")"):
                    correct = raw_answer.strip() == ref_raw_answer[len(ref_answer) + 2 :].strip()
                else:
                    correct = raw_answer.strip() == ref_raw_answer.strip()
            parsed_answer = raw_answer
    elif question_type == "short-answer":
        correct = ref_answer.lower() == answer.lower()
        parsed_answer = answer

    return {
        "raw_answer": raw_answer,
        "parsed_answer": parsed_answer,
        "correct": correct,
    }


def mantis_process_results(doc, results):
    pred = results[0]
    question_type, answer, options = doc["question_type"], doc["answer"], doc["options"]

    parsed_pred = get_prediction(question_type, pred, answer, options)
    data_dict = {
        "question_id": doc["id"],
        "pred_answer": parsed_pred["parsed_answer"],
        "answer": doc["answer"],
        "correct": parsed_pred["correct"],
    }

    return {"mantis_score": data_dict}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:
        if gold_i == pred_i:
            correct = True
    return correct


def mantis_aggregation(results):
    score = 0
    for result in results:
        if result["correct"]:
            score += 1
    avg_score = score / len(results)

    return avg_score