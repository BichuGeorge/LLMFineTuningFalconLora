from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import torch
"""
Finetune a Falcon model with LoRA on IMDb dataset for casual inferencing (Like chatgpt). So, not using the labels for training.
"""

"""
ABOUT THE MODEL:

Training Details
Training Data
Falcon-RW-1B was trained on 350B tokens of RefinedWeb, a high-quality filtered and deduplicated web dataset. The data was tokenized with the GPT-2 tokenizer.

Training Procedure
Falcon-RW-1B was trained on 32 A100 40GB GPUs, using only data parallelism with ZeRO.

Training Hyperparameters
Hyperparameters were adapted from the GPT-3 paper (Brown et al., 2020).

Hyperparameter	Value	Comment
Precision	bfloat16	
Optimizer	AdamW	
Learning rate	2e-4	500M tokens warm-up, cosine decay to 2e-5
Weight decay	1e-1	
Batch size	512	4B tokens ramp-up
Speeds, Sizes, Times
Training happened in early December 2022 and took about six days.

Evaluation
See the 📓 paper on arXiv for in-depth evaluation.

Technical Specifications
Model Architecture and Objective
Falcon-RW-1B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

The architecture is adapted from the GPT-3 paper (Brown et al., 2020), but uses ALiBi (Ofir et al., 2021) and FlashAttention (Dao et al., 2022).
"""

"""
1. Load the model and tokenizer.
2. Load the dataset.
3. Tokenize the dataset using the same tokenizer.
4. Prepare the model for specific task (here for casual inference, so that we only take the text and not the label)
5. Set the parameters for LoRA (Low-Rank Adaptation).
6. Apply LoRA to the model.

"""
# Step 1: Model & tokenizer
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

# Step 2: Load dataset (IMDb for sentiment classification)
dataset = load_dataset("imdb")

# Step 3: Tokenization function
def tokenize(batch):
    tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128) # Whatever data that we take, we are going to convert them into token length of 128
    tokens["labels"] = tokens["input_ids"]  # required for causal LM loss
    return tokens


tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},  # use CUDA if available, else CPU
    trust_remote_code=True,
    offload_folder="falcon_offload"
)

# Step 5: Prepare model for 8-bit training
model = prepare_model_for_kbit_training(model)

# Step 6: Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    target_modules=["query_key_value"]  # Falcon-specific
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 7: TrainingArguments
training_args = TrainingArguments(
    output_dir="falcon_lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_total_limit=1,
    fp16=False,
    logging_dir="./logs",
    report_to="none",
    no_cuda=True
)

# Step 8: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].select(range(1000))  # small subset for testing
)

# Step 9: Train
trainer.train()
