#Source: https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/utils_qa.py
"""
Utility functions for switch precision training
"""

import os
import json
import logging
import torch
import collections
import json
import logging
import os
from typing import Optional
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from models.modeling_gpt2_quant import GPT2ForQuestionAnswering as GPT2ForQuestionAnsweringModel
from transformers import GPT2ForQuestionAnswering
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def count_trainable_parameters(model):
    """Count and log the number of trainable parameters"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logging.info(f"Total parameters: {total_params:,}")
    print(f"Total parameters: {total_params:,}")

def get_lora_params(model, w_bits):
    """
    Get parameters specific to LoRA modules for given bit-widths.
    
    Args:
        model: The model containing QuantizeLoraLinear modules
        w_bits (list): List of bit-widths to get parameters for
        
    Returns:
        list: Parameters for the specified bit-width LoRA modules
    """
    from models.quantized_lora_linear import QuantizeLoraLinear
    
    lora_params = []
    for num_bits in w_bits:
        lora_key = f'adapter_{num_bits}'
        
        for name, module in model.named_modules():
            if isinstance(module, QuantizeLoraLinear):
                module.layer_name = name
                if lora_key in module.lora_adapters:
                    lora_params.extend(module.lora_adapters[lora_key].parameters())  
                    print(f"Found LoRA parameters for {name}")
    return lora_params


def freeze_model_for_lora_training(model, config):
    """
    Freeze base model parameters and unfreeze only LoRA parameters (like run_qa.py)
    """
    # Freeze the base model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the LoRA parameters for all bit-widths
    lora_params = get_lora_params(model, config.w_bits)
    for param in lora_params:
        #print(f"Unfreezing: {param.name}")
        param.requires_grad = True
        
    # Unfreeze the qa_outputs layer
    if hasattr(model, 'module'):  # DataParallel case
        model.module.qa_outputs.weight.requires_grad = True
        model.module.qa_outputs.bias.requires_grad = True
    else:  # Single GPU case
        model.qa_outputs.weight.requires_grad = True
        model.qa_outputs.bias.requires_grad = True
    
    # Count and log trainable parameters
    count_trainable_parameters(model)


def save_model(model, model_path):
    """Save model checkpoint - handles DataParallel correctly"""
    # Handle DataParallel case
    if hasattr(model, 'module'):
        # DataParallel case - save the underlying model
        torch.save(model.module.state_dict(), model_path)
    else:
        # Single GPU case
        torch.save(model.state_dict(), model_path)
    logging.info("Model saved to %s", model_path)


def save_training_state(model, optimizer, lr_policy, epoch, global_step, metrics, output_dir):
    """Save training state including model, optimizer, and metrics - handles DataParallel correctly"""
    # Handle DataParallel case for model state
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_policy_state_dict': lr_policy.state_dict() if hasattr(lr_policy, 'state_dict') else None,
        'epoch': epoch,
        'global_step': global_step,
        'metrics': metrics,
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint-{global_step}')
    os.makedirs(checkpoint_path, exist_ok=True)
    
    torch.save(checkpoint, os.path.join(checkpoint_path, 'pytorch_model.bin'))
    logging.info("Training state saved to %s", checkpoint_path)
    
    return checkpoint_path

def load_model(config):
    
    model = GPT2ForQuestionAnsweringModel(config,)
    base_model = GPT2ForQuestionAnswering.from_pretrained("gpt2")

    model_sd = model.state_dict()
    base_model_sd = base_model.state_dict()

    for key in base_model_sd.keys():
        if key not in model_sd:
            # Some HF keys might not exist in the custom model (e.g., heads)
            print(f"[Skip] Key '{key}' not found in custom model.")
            continue
        if any(key.endswith(w) for w in ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']):
            with torch.no_grad():
                model_sd[key].copy_(base_model_sd[key].t())
        else:
            with torch.no_grad():
                model_sd[key].copy_(base_model_sd[key])
    
    return model


class QADistillationLoss(nn.Module):
    """
    Distillation loss module for QA models (start & end logits).
    Computes KL divergence between student and teacher outputs.
    """
    def __init__(self, temperature=2.0, reduction='batchmean'):
        super(QADistillationLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.kl_loss_fn = nn.KLDivLoss(reduction=reduction)
    
    def forward(self, student_start, student_end, teacher_start, teacher_end):
        """
        Args:
            student_start, student_end: [batch_size, seq_len] logits from student
            teacher_start, teacher_end: [batch_size, seq_len] logits from teacher
        Returns:
            avg_distill_loss: average distillation loss over start and end logits
        """
        T = self.temperature
        
        # log-probabilities for student
        student_start_log = F.log_softmax(student_start / T, dim=-1)
        student_end_log   = F.log_softmax(student_end / T, dim=-1)
        
        # probabilities for teacher
        teacher_start_prob = F.softmax(teacher_start / T, dim=-1)
        teacher_end_prob   = F.softmax(teacher_end / T, dim=-1)
        
        # KL divergence
        start_loss = self.kl_loss_fn(student_start_log, teacher_start_prob) * (T ** 2)
        end_loss   = self.kl_loss_fn(student_end_log, teacher_end_prob) * (T ** 2)
        
        # average
        avg_distill_loss = (start_loss + end_loss) / 2.0
        return avg_distill_loss




def load_training_state(model, optimizer, lr_policy, checkpoint_path):
    """Load training state from checkpoint - handles DataParallel correctly"""
    checkpoint = torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin'))
    
    # Handle DataParallel case for model loading
    if hasattr(model, 'module'):
        # DataParallel case - load into the underlying model
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Single GPU case
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if checkpoint['lr_policy_state_dict'] is not None and hasattr(lr_policy, 'load_state_dict'):
        lr_policy.load_state_dict(checkpoint['lr_policy_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    metrics = checkpoint.get('metrics', {})
    
    logging.info("Training state loaded from %s", checkpoint_path)
    logging.info("Resuming from epoch %d, global step %d", epoch, global_step)
    
    return epoch, global_step, metrics


def save_metrics_to_json(metrics, output_dir, prefix=""):
    """Save metrics to JSON file (like run_qa.py)"""
    import json
    
    metrics_file = os.path.join(output_dir, f"{prefix}_results.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info("Metrics saved to %s", metrics_file)


def save_tokenizer(tokenizer, output_dir):
    """Save tokenizer (like run_qa.py)"""
    tokenizer.save_pretrained(output_dir)
    logging.info("Tokenizer saved to %s", output_dir)

def postprocess_qa_predictions(
    examples,
    features,
    predictions: tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if (
            version_2_with_negative
            and min_null_prediction is not None
            and not any(p["offsets"] == (0, 0) for p in predictions)
        ):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise OSError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions
