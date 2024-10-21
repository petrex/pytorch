import collections
import datasets
import evaluate
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.ao.pruning import WeightNormSparsifier
import transformers

# force CUTLASS use if cuSPARSELt is not available
SparseSemiStructuredTensor._FORCE_CUTLASS = True
torch.manual_seed(100)

def preprocess_validation_function(examples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def preprocess_train_function(examples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, (offset, answer) in enumerate(zip(offset_mapping, answers)):
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30
    metric = evaluate.load("squad")

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    # for example in tqdm(examples):
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0
                    # or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def measure_execution_time(model, batch_sizes, dataset):
    dataset_for_model = dataset.remove_columns(["example_id", "offset_mapping"])
    dataset_for_model.set_format("torch")
    model.cuda()
    batch_size_to_time_sec = {}
    for batch_size in batch_sizes:
        batch = {
            k: dataset_for_model[k][:batch_size].to(model.device)
            for k in dataset_for_model.column_names
        }

        with torch.inference_mode():
            timer = benchmark.Timer(
                stmt="model(**batch)", globals={"model": model, "batch": batch}
            )
            p50 = timer.blocked_autorange().median * 1000
        batch_size_to_time_sec[batch_size] = p50
    return batch_size_to_time_sec

# load model
model_name = "bert-base-cased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
print(f"Loading tokenizer: {model_name}")
print(f"Loading model: {model_name}")

# set up train and val dataset
squad_dataset = datasets.load_dataset("squad")
tokenized_squad_dataset = {}
tokenized_squad_dataset["train"] = squad_dataset["train"].map(
    lambda x: preprocess_train_function(x, tokenizer), batched=True
)
tokenized_squad_dataset["validation"] = squad_dataset["validation"].map(
    lambda x: preprocess_validation_function(x, tokenizer),
    batched=True,
    remove_columns=squad_dataset["train"].column_names,
)
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

training_args = transformers.TrainingArguments(
    "trainer",
    num_train_epochs=1,
    lr_scheduler_type="constant",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=512,
)

trainer = transformers.Trainer(
    model,
    training_args,
    train_dataset=tokenized_squad_dataset["train"],
    eval_dataset=tokenized_squad_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# batch sizes to compare for eval
batch_sizes = [4, 16, 64, 256]
# 2:4 sparsity require fp16, so we cast here for a fair comparison
with torch.autocast("cuda"):
    with torch.inference_mode():
        predictions = trainer.predict(tokenized_squad_dataset["validation"])
    start_logits, end_logits = predictions.predictions
    fp16_baseline = compute_metrics(
        start_logits,
        end_logits,
        tokenized_squad_dataset["validation"],
        squad_dataset["validation"],
    )
    fp16_time = measure_execution_time(
        model,
        batch_sizes,
        tokenized_squad_dataset["validation"],
    )
print("fp16", fp16_baseline)
print("cuda_fp16 time", fp16_time)

# fp16 {'exact_match': 78.53358561967833, 'f1': 86.9280493093186}
# cuda_fp16 time {4: 10.927572380751371, 16: 19.607915310189128, 64: 73.18846387788653, 256: 286.91255673766136}

sparsifier = WeightNormSparsifier(
    # apply sparsity to all blocks
    sparsity_level=1.0,
    # shape of 4 elemens is a block
    sparse_block_shape=(1, 4),
    # two zeros for every block of 4
    zeros_per_block=2
)

# add to config if nn.Linear and in the BERT model.
sparse_config = [
    {"tensor_fqn": f"{fqn}.weight"}
    for fqn, module in model.named_modules()
    if isinstance(module, nn.Linear) and "layer" in fqn
]

# Prepare the model, insert fake-sparsity parameterizations for training
sparsifier.prepare(model, sparse_config)
print(model.bert.encoder.layer[0].output)

# BertOutput(
#   (dense): ParametrizedLinear(
#     in_features=3072, out_features=768, bias=True
#     (parametrizations): ModuleDict(
#       (weight): ParametrizationList(
#         (0-5): 6 x FakeSparsity()
#       )
#     )
#   )
#   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#   (dropout): Dropout(p=0.1, inplace=False)
# )

sparsifier.step()
with torch.autocast("cuda"):
    with torch.inference_mode():
        predictions = trainer.predict(tokenized_squad_dataset["validation"])
    pruned = compute_metrics(
        *predictions.predictions,
        tokenized_squad_dataset["validation"],
        squad_dataset["validation"],
    )
print("pruned eval metrics:", pruned)
# pruned eval metrics: {'exact_match': 40.59602649006622, 'f1': 56.51610004515979}

trainer.train()
sparsifier.squash_mask()
torch.set_printoptions(edgeitems=4)
print(model.bert.encoder.layer[0].intermediate.dense.weight)

# Parameter containing:
# tensor([[ 0.0000, -0.0237,  0.0000,  0.0130,  ..., -0.0462, -0.0000, 0.0000, -0.0272],
#        [ 0.0436, -0.0000, -0.0000,  0.0492,  ..., -0.0000,  0.0844,  0.0340, -0.0000],
#        [-0.0302, -0.0350,  0.0000,  0.0000,  ...,  0.0303,  0.0175, -0.0000,  0.0000],
#        [ 0.0000, -0.0000, -0.0529,  0.0327,  ...,  0.0213,  0.0000, -0.0000,  0.0735],
#        ...,
#        [ 0.0000, -0.0000, -0.0258, -0.0239,  ..., -0.0000, -0.0000,  0.0380,  0.0562],
#        [-0.0432, -0.0000,  0.0000, -0.0598,  ...,  0.0000, -0.0000,  0.0262  -0.0227],
#        [ 0.0244,  0.0921, -0.0000, -0.0000,  ..., -0.0000, -0.0784,  0.0000,  0.0761],
#        [ 0.0000,  0.0225, -0.0395, -0.0000,  ..., -0.0000,  0.0684, -0.0344, -0.0000]], device='cuda:0', requires_grad=True)

model = model.cuda().half()
# accelerate for sparsity
for fqn, module in model.named_modules():
    if isinstance(module, nn.Linear) and "layer" in fqn:
        module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))

with torch.inference_mode():
    predictions = trainer.predict(tokenized_squad_dataset["validation"])
start_logits, end_logits = predictions.predictions
metrics_sparse = compute_metrics(
    start_logits,
    end_logits,
    tokenized_squad_dataset["validation"],
    squad_dataset["validation"],
)
print("sparse eval metrics: ", metrics_sparse)
sparse_perf = measure_execution_time(
    model,
    batch_sizes,
    tokenized_squad_dataset["validation"],
)
print("sparse perf metrics: ", sparse_perf)

# sparse eval metrics:  {'exact_match': 78.43897824030275, 'f1': 86.48718950090766}
# sparse perf metrics:  {4: 12.621004460379481, 16: 15.368514601141214, 64: 58.702805917710066, 256: 244.19364519417286}