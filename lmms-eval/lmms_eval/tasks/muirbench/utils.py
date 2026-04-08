import re

import pandas as pd

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter


def muir_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"


def muir_doc_to_visual(doc):
    image_list = [image.convert("RGB") for image in doc["image_list"]]
    return image_list


def muir_doc_to_target(doc):
    return doc["answer"]


def muir_process_results(doc, result):
    pred = result[0]
    task = doc["task"]
    idx = doc["idx"]
    image_relation = doc["image_relation"]
    answer = doc["answer"]
    image_type = doc["image_type"]

    data_dict = {
        "pred": pred,
        "task": task,
        "idx": idx,
        "image_relation": image_relation,
        "answer": answer,
        "image_type": image_type,
    }

    return {"muirbench_score_overall": data_dict}


def muir_aggregation(results):
    task_num = {}
    score = 0
    task_score = {}
    for result in results:
        if result["task"] not in task_score:
            task_score[result["task"]] = 0

        if result["task"] not in task_num:
            task_num[result["task"]] = 0

        if result["pred"].lower().strip() == result["answer"].lower().strip():
            task_score[result["task"]] += 1
            score += 1
        task_num[result["task"]] += 1

    score = score / len(results)
    task_score = {k: v / task_num[k] for k, v in task_score.items()}

    print("=" * 50)
    for k, v in task_score.items():
        print(f"{k} : {v:.2f}")
    print("=" * 50)
    return score


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps


# import re
# from functools import lru_cache

# import pandas as pd
# from datasets import load_dataset

# from lmms_eval.filters.extraction import ExtendedRegexFilter
# from lmms_eval.filters.transformation import MapFilter


# def muir_doc_to_text(doc, lmms_eval_specific_kwargs=None):
#     question, choices = doc["question"], doc["options"]
#     len_choices = len(choices)
#     post_prompt = lmms_eval_specific_kwargs["post_prompt"]
#     pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
#     options = [chr(ord("A") + i) for i in range(len_choices)]
#     choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
#     return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"


# def muir_doc_to_visual(doc):
#     image_list = [image.convert("RGB") for image in doc["image_list"]]
#     return image_list


# def muir_doc_to_target(doc):
#     return doc["answer"]


# @lru_cache(maxsize=1)
# def _muir_idx_to_num_images():
#     """
#     Build a cache: sample idx -> number of input images
#     using the original HF test split.
#     This is needed because lmms-eval may drop image columns
#     before passing doc into process_results().
#     """
#     ds = load_dataset("MUIRBENCH/MUIRBENCH", split="test", token=True)

#     idx2n = {}
#     for sample in ds:
#         sample_idx = str(sample["idx"])
#         if "image_list" not in sample or sample["image_list"] is None:
#             raise KeyError(f"Raw dataset sample idx={sample_idx} does not contain 'image_list'.")
#         idx2n[sample_idx] = len(sample["image_list"])

#     print(f"[MUIR] Built idx->num_images cache for {len(idx2n)} samples.")
#     return idx2n


# def muir_get_num_images(doc):
#     """
#     Prefer reading directly from doc if image fields exist.
#     Otherwise, fall back to idx-based lookup from the original HF dataset.
#     """
#     # direct path: if process_results doc still contains image fields
#     if "image_list" in doc and doc["image_list"] is not None:
#         return len(doc["image_list"])
#     if "image" in doc and doc["image"] is not None:
#         if isinstance(doc["image"], (list, tuple)):
#             return len(doc["image"])
#         return 1
#     if "images" in doc and doc["images"] is not None:
#         if isinstance(doc["images"], (list, tuple)):
#             return len(doc["images"])
#         return 1

#     # fallback path: lookup by idx
#     if "idx" not in doc:
#         raise KeyError(f"Cannot infer num_images because doc has no image field and no idx. Keys: {list(doc.keys())}")

#     idx_str = str(doc["idx"])
#     idx2n = _muir_idx_to_num_images()

#     if idx_str not in idx2n:
#         raise KeyError(f"idx={idx_str} not found in cached HF dataset mapping.")

#     return idx2n[idx_str]


# def muir_process_results(doc, result):
#     pred = result[0]
#     task = doc.get("task", "unknown_task")
#     idx = doc.get("idx", -1)
#     image_relation = doc.get("image_relation", "unknown_relation")
#     answer = doc["answer"]
#     image_type = doc.get("image_type", "unknown_type")

#     num_images = muir_get_num_images(doc)

#     data_dict = {
#         "pred": pred,
#         "task": task,
#         "idx": idx,
#         "image_relation": image_relation,
#         "answer": answer,
#         "image_type": image_type,
#         "num_images": num_images,
#     }

#     metrics = {
#         "muirbench_score_overall": data_dict,
#     }

#     if num_images == 2:
#         metrics["muirbench_score_img2"] = data_dict
#     elif num_images == 4:
#         metrics["muirbench_score_img4"] = data_dict
#     elif num_images == 8:
#         metrics["muirbench_score_img8"] = data_dict

#     return metrics


# def muir_aggregation(results):
#     if len(results) == 0:
#         print("=" * 50)
#         print("Empty group encountered. Return 0.0")
#         print("=" * 50)
#         return 0.0

#     task_num = {}
#     score = 0
#     task_score = {}

#     # optional: print current group's num_images composition
#     num_image_stat = {}

#     for result in results:
#         if result["task"] not in task_score:
#             task_score[result["task"]] = 0
#         if result["task"] not in task_num:
#             task_num[result["task"]] = 0

#         nimg = result.get("num_images", None)
#         if nimg is not None:
#             num_image_stat[nimg] = num_image_stat.get(nimg, 0) + 1

#         if str(result["pred"]).lower().strip() == str(result["answer"]).lower().strip():
#             task_score[result["task"]] += 1
#             score += 1
#         task_num[result["task"]] += 1

#     score = score / len(results)
#     task_score = {k: v / task_num[k] for k, v in task_score.items()}

#     print("=" * 50)
#     print(f"Group sample count: {len(results)}")
#     if len(num_image_stat) > 0:
#         print("Num-images distribution:", dict(sorted(num_image_stat.items(), key=lambda x: x[0])))
#     for k, v in task_score.items():
#         print(f"{k} : {v:.2f}")
#     print("=" * 50)
#     return score


# class MultiChoiceRegexFilter(ExtendedRegexFilter):
#     def __init__(self, *args, **kwargs):
#         """
#         regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
#                         - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
#                         - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
#         group_select: Selects the (group_select)th match from the findall result.
#         ignore_case: Ignores the case during step 1 matching
#         ignore_punctuation: Remove the punctuation during step 1 matching
#         regexes_to_ignore: Remove these regexes during step 1 matching
#         """
#         super().__init__(*args, **kwargs)

#     def apply(self, resps, docs):
#         filtered_resps = []

#         for r, doc in zip(resps, docs):
#             option_letter_regex = re.compile(r"^\s*([A-Z])\.")

#             filtered = []
#             for resp in r:
#                 match = option_letter_regex.match(resp)
#                 if match:
#                     filtered.append(match.group(1))
#                 else:
#                     filtered.append(resp)

#             filtered_resps.append(filtered[0])

#         return filtered_resps