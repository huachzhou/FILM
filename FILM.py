import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss, MSELoss

class FILM(LlamaPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        
        super().__init__(config, *inputs, **kwargs)
        self.model = LlamaModel(config)

        self.dropout = 0.2
        self.tau = 1.2
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
                if self.task == 'temperature_pretrain':
                    pass
                else:
                    labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
                    labels_index_row = labels_index[:, 0]
                    labels_index_row = torch.cat(((labels_index_row[:-1] != labels_index_row[1:]).nonzero(), torch.tensor(labels_index_row.shape[0] - 1).to(labels_index_row.device).unsqueeze(0).unsqueeze(1))).squeeze()
                    labels_index = labels_index[labels_index_row]
                    w = torch.nn.functional.normalize(hidden_states[labels_index[:, 0], labels_index[:, 1] - 1].reshape(-1, 4096))  
                    if self.training:
                        context = self.temp(torch.nn.functional.dropout(torch.nn.functional.normalize(hidden_states.mean(dim=1)), p = self.dropout))
                    else:
                        context = self.temp(torch.nn.functional.normalize(hidden_states.mean(dim=1)))
                    
                    pos_ratings = torch.sum(w.detach() * context, dim=-1)  
                    tot_ratings = torch.matmul(w.detach(), torch.transpose(context, 0, 1))
                    sm_temp =  (pos_ratings / tot_ratings.sum(dim=-1)) * w.shape[0]     
                    sm_temp = torch.clamp(sm_temp, 0, self.tau)
                    logits = logits * sm_temp[:,None,None]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                shift_labels = torch.where(shift_labels==0, -100, shift_labels)
                loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    



        





