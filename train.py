import os
import torch
import transformers
from datasets import load_dataset
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from FILM import FILM
import fire

class FILM_Callback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):  
        if state.epoch >= 3:
            control.should_training_stop = True
        return control

class FILMTrainer(Trainer):
    def __int__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)        
    def compute_discrimn_loss_empirical(self, W):
        m, p = W.shape
        I = torch.eye(p).to(W.device)
        scalar = p / (m * 0.01)
        logdet = torch.logdet(I + scalar * (W.T).matmul(W))
        return logdet / 2.
        
    def compute_discrimn_loss_group(self, W, Pi):
        m, p = W.shape
        I = torch.eye(p).to(W.device)
        compress_loss = 0.
        for j in range(2):
            trPi = torch.sum(Pi[j]) + 1e-8
            scalar = p / (trPi * 0.01)
            a = (W.T) * Pi[j].view(1, -1)
            log_det = torch.logdet(I + scalar * a.matmul(W))
            compress_loss += log_det * trPi / (2 * m)
        return compress_loss
        
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs["input_ids"][:,-2] = 0
        inputs["labels"][:,:-2] = 0

        if model.base_model.model.task =='temperature_train':
            if not return_outputs:
                labels_index = torch.argwhere(torch.bitwise_or(inputs["labels"] == 8241, inputs["labels"] == 3782))
                labels_index_row = labels_index[:, 0]
                labels_index_row = torch.cat(((labels_index_row[:-1] != labels_index_row[1:]).nonzero(), 
                                             torch.tensor(labels_index_row.shape[0] - 1).to(labels_index_row.device).unsqueeze(0).unsqueeze(1))).squeeze()

                labels_index = labels_index[labels_index_row]
                embed = model.base_model.model.model.embed_tokens(inputs['input_ids'])

                inputs['inputs_embeds'] = embed
                inputs['input_ids'] = None
                outputs = model(**inputs, output_hidden_states=True)

                logits = outputs.logits
                logits = logits.softmax(dim=-1)
                logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1] - 1][:,[3782, 8241]], dim=-1)

                w = torch.nn.functional.normalize(outputs["hidden_states"][-1][labels_index[:, 0], labels_index[:, 1] - 1].reshape(-1, 4096))

                loss_all = self.compute_discrimn_loss_empirical(w)
                loss_group = self.compute_discrimn_loss_group(w, [logits[:,0], logits[:, 1]])

                return outputs.loss + 0.01 * (loss_all - loss_group)
                
            if return_outputs:
                outputs = model(**inputs, output_hidden_states=False)
                return outputs.loss, outputs
        else:
            inputs["labels"] = inputs["input_ids"]
            outputs = model(**inputs)
            if not return_outputs:
                return outputs.loss 
            if return_outputs:
                return outputs.loss, outputs

class BaseTrainer(Trainer): 
    def __int__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)       
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs["input_ids"][:,-2] = 0
        inputs["labels"][:,:-2] = 0
        outputs = model(**inputs)
        if not return_outputs:
            return outputs.loss 
        if return_outputs:
            return outputs.loss, outputs

def configure_lora_model(
    model,
    lora_r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["temp"],
    checkpoint_path=None,
):
    # 创建LoRA配置
    model.temp = torch.nn.Linear(4096, 4096, bias=False).to(model.device)
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    checkpoint_path = os.path.join(checkpoint_path, "pytorch_model.bin")  
    adapters_weights = torch.load(checkpoint_path)
    set_peft_model_state_dict(model, adapters_weights)
    return model


def compute_metrics(eval_preds):
    pre, labels = eval_preds
    auc = roc_auc_score(pre[1], pre[0])
    logloss = log_loss(pre[1], pre[0])
    print({'auc': auc, 'logloss': logloss})
    return {'auc': auc, 'logloss': logloss}
   
def preprocess_logits_for_metrics(logits, labels):
    labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
    labels_index_row = labels_index[:, 0]
    labels_index_row = torch.cat(((labels_index_row[:-1] != labels_index_row[1:]).nonzero(), torch.tensor(labels_index_row.shape[0] - 1).to(labels_index_row.device).unsqueeze(0).unsqueeze(1))).squeeze()
    labels_index = labels_index[labels_index_row]
    if len(labels_index.shape) == 1:
        labels_index = labels_index.unsqueeze(0)
    labels_ = labels[labels_index[:, 0], labels_index[:, 1]]
    labels_ = torch.where(labels_==3782, 0, 1)
    logits_ = logits.softmax(dim=-1)
    logits_ = torch.softmax(logits[[labels_index[:, 0], labels_index[:, 1] - 1]][:,[3782, 8241]], dim = -1)[:,-1]
    return logits_, labels_

def compute_metrics_temp_pretrain(eval_preds):
    pre, labels = eval_preds
    pre, labels = pre
    acc = (pre == labels).mean()
    return {'auc': acc}
    
def preprocess_logits_for_temp_pretrain_metrics(logits, labels):

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().flatten()
    pre = torch.argmax(shift_logits, dim=2).flatten()
    return pre, shift_labels
            

def train(
    # model/data params
    base_model: str = "",
    model_type: str = "",
    train_data_path: str = "",
    val_data_path: str = "",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    train_batch_size: int = 32,
    test_batch_size: int = 72,
    num_epochs: int = 40,
    learning_rate: float = 1e-4,
    cutoff_len: int = 1024,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.2,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
):
    
    cudnn.benchmark = True
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["WANDB_DISABLED"] = "true"
    device_map = "auto"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if model_type == 'Base':
        model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,
        )
    else:
        model = FILM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
    )


   
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    train_data = load_dataset("json", data_files=train_data_path)
    val_data = load_dataset("json", data_files=val_data_path)
    train_data["train"] = train_data["train"].shuffle(seed=2023)

    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))


    
    eval_step = len(train_data) // train_batch_size + 1

    train_args = transformers.TrainingArguments(
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=test_batch_size,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            group_by_length=group_by_length,
    )

    if model_type == 'Base':
        model = configure_lora_model(
            model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            modules_to_save=["temp"],
            checkpoint_path=resume_from_checkpoint
        )
        trainer = BaseTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )
        trainer.train(resume_from_checkpoint=False)
    else:
        model = configure_lora_model(
            model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            modules_to_save=["temp",'lm_head'],
            checkpoint_path=resume_from_checkpoint
        )
        trainer = FILMTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            compute_metrics=compute_metrics_temp_pretrain,
            preprocess_logits_for_metrics=preprocess_logits_for_temp_pretrain_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10), FILM_Callback]
        )
        model.base_model.model.task = 'temperature_pretrain'
        trainer.train(resume_from_checkpoint=False)

        model = FILM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        model.temp = torch.nn.Linear(4096, 4096, bias=False).to(model.device)
        model = PeftModel.from_pretrained(model, './output/checkpoint-3936/')
        model = model.merge_and_unload()
        model = configure_lora_model(
            model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            modules_to_save=["temp"],
            checkpoint_path=resume_from_checkpoint
        )

        trainer = FILMTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )
        model.base_model.model.task = 'temperature_train'
        trainer.train(resume_from_checkpoint=False)
def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

if __name__ == "__main__":
    fire.Fire(train)
