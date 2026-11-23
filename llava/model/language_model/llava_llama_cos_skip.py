"""Skip-FFN LLaVA model for the SV-CoS setting (Nr=272, Ns=256).

The Chain-of-Sight resampler produces 16 global + 256 mid + 1024 local query
tokens. The local stream is merged down to 256 tokens; the 272 retained
tokens (16 global + 256 mid) pass through the FFN, and the 256 merged local
tokens bypass it. The last local visual token also passes through the FFN
(the LV component in the paper's ablation).
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch_index_q_tome import LlavaMetaModel, LlavaMetaForCausalLM
from .llava_llama_cos_skip_ffn import SkipFFNLlamaForCausalLM, SkipFFNLlamaModel


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaCoSSkipModel(LlavaMetaModel, SkipFFNLlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaCoSSkipModel, self).__init__(config)


class LlavaLlamaCoSSkipForCausalLM(SkipFFNLlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(SkipFFNLlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaCoSSkipModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )
        _inputs["images"] = images
        _inputs["image_sizes"] = image_sizes
        return _inputs
