"""Load the chat model the same way for eval and serving.

On CUDA (Colab T4) the base model is loaded in 4-bit NF4 with fp16 compute; the T4 has
no bf16 and can't hold an 8B model in fp16. The adapter (if any) is applied on top of
that same 4-bit base, which is also how QLoRA trains it. Decoding is greedy so a rerun
gives the same answers.
"""
import torch

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


class Generator:
    def __init__(self, model_id: str = LLAMA, adapter: str | None = None, token: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id, self.adapter = model_id, adapter
        self.cuda = torch.cuda.is_available()
        kwargs = {"token": token}
        if self.cuda:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = "auto"
            kwargs["dtype"] = torch.float16
        self.quantization = "4bit-nf4" if self.cuda else "none-cpu"
        # With an adapter, load the tokenizer saved next to it: the exact one used in training.
        self.tok = AutoTokenizer.from_pretrained(adapter or model_id, token=token)
        if self.tok.pad_token_id is None:  # Llama ships without a pad token; generate() needs one
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, messages: list[dict], max_new_tokens: int = 400) -> str:
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        # Greedy (do_sample=False): the same question always gets the same answer, so scores are reproducible.
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        # decode only the new tokens, not the prompt
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
