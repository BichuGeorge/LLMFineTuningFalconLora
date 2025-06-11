# 🦅 LLM Finetuning: Falcon-RW-1B for Causal Inference using LoRA

This project demonstrates how to finetune the [`tiiuae/falcon-rw-1b`](https://huggingface.co/tiiuae/falcon-rw-1b) language model for **causal inference** using **LoRA (Low-Rank Adaptation)**. The IMDb dataset is used for training, but **only the raw text is used (labels are ignored)** since the task is to train the model for causal (next-token) prediction, not classification.

🔗 Finetuned model on Hugging Face: [BichuGeo/falcon-lora-imdb-causual-inference](https://huggingface.co/BichuGeo/falcon-lora-imdb-causual-inference)

---

## 📌 Objective

To adapt a lightweight version of Falcon-RW-1B using LoRA for causal language modeling. This approach is efficient and well-suited for scenarios with limited compute resources.

---

## 📚 Dataset

- **IMDb Dataset**  
  - Source: Hugging Face Datasets  
  - Usage: Only the `text` field is used for training. The sentiment `label` is ignored.
  - Task: Language modeling (causal inference / next token prediction)

---

## 🛠️ Steps

1. **Load the Base Model and Tokenizer**
   - `tiiuae/falcon-rw-1b` model is loaded using Hugging Face Transformers.

2. **Load IMDb Dataset**
   - The dataset is loaded via the Hugging Face `datasets` library.

3. **Tokenize the Dataset**
   - The raw text is tokenized using the same tokenizer as the base model.
   - Padding and truncation are handled appropriately.

4. **Prepare for Causal Language Modeling**
   - The dataset is formatted to return only the input tokens (`input_ids`) without any classification labels.
   - This ensures it's suitable for next-token prediction.

5. **Set Up LoRA Parameters**
   - LoRA configuration is set up (e.g., `r`, `alpha`, `dropout`, and target modules).
   - LoRA is applied to specific transformer layers to reduce the number of trainable parameters.

6. **Apply LoRA to the Model**
   - The model is adapted with LoRA using the `peft` library (or similar).
   - Finetuning is performed with appropriate training hyperparameters.

7. **Push to Hugging Face**
   - The finetuned model is uploaded to Hugging Face Model Hub under:  
     👉 [BichuGeo/falcon-lora-imdb-causual-inference](https://huggingface.co/BichuGeo/falcon-lora-imdb-causual-inference)

---

## 🚀 Results

The resulting model can perform causal inference on raw English text (e.g., predicting the next word given a context) and is a lightweight alternative to full finetuning thanks to LoRA.

---

## 💡 Technologies Used

- Hugging Face Transformers
- Hugging Face Datasets
- LoRA (`peft` or `trl` library)
- PyTorch

---
