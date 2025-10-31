#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from torch.nn import CrossEntropyLoss

from ..llava_arch_vis_supervision import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfigWithVisualSupervision(LlamaConfig):
    model_type = "llava_llama_with_visual_supervision"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfigWithVisualSupervision

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaVisSupervisionForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfigWithVisualSupervision

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # import pdb; pdb.set_trace()
        self.post_init()

    def get_model(self):
        return self.model

    def initialize_vision_head(self):
        # import pdb; pdb.set_trace()
        assert hasattr(self.model, "vqkd_model")
        self.vision_head = nn.Linear(self.config.hidden_size, self.model.vqkd_model.n_embed, bias=False)
        self._init_weights(self.vision_head)

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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

        # import pdb; pdb.set_trace()

        logits_lm = self.lm_head(hidden_states)
        logits_vis = self.vision_head(hidden_states)

        logits_lm = logits_lm.float()
        logits_vis = logits_vis.float()

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits_lm = logits_lm[..., :-1, :].contiguous()
            shift_logits_vis = logits_vis[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits_lm = shift_logits_lm[shift_labels < self.model.embed_tokens.num_embeddings]
            shift_logits_vis = shift_logits_vis[shift_labels > self.model.embed_tokens.num_embeddings]

            shift_labels_lm = shift_labels[shift_labels < self.model.embed_tokens.num_embeddings]
            shift_labels_lm = shift_labels_lm.to(shift_logits_lm.device)

            shift_labels_vis = shift_labels[shift_labels > self.model.embed_tokens.num_embeddings] - self.model.embed_tokens.num_embeddings
            shift_labels_vis = shift_labels_vis.to(shift_logits_vis.device)

            loss_lm = loss_fct(shift_logits_lm, shift_labels_lm)
            loss_vis = loss_fct(shift_logits_vis, shift_labels_vis)
            loss = loss_lm + loss_vis

        if not return_dict:
            output = ((logits_lm, logits_vis),) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=(logits_lm, logits_vis),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama_with_visual_supervision", LlavaConfigWithVisualSupervision)
AutoModelForCausalLM.register(LlavaConfigWithVisualSupervision, LlavaLlamaVisSupervisionForCausalLM)

