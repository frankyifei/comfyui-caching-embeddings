import torch
from collections import OrderedDict

torch.backends.cuda.matmul.allow_tf32 = True

class CLIPTextEncodeWithCache:

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def __init__(self):
        self.cache = OrderedDict()

    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "cache_size":("INT", {"default": 20, "min": 1, "max": 100}),
                "clip": ("CLIP",)}
        }

    def encode(
        self, clip: torch.nn.Module, text: str, cache_size:int
    ) -> tuple[list[list[torch.Tensor, dict[str, torch.Tensor]]]]:
        if text not in self.cache:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            if len(self.cache) >= cache_size:
                self.cache.popitem(last=False)
            self.cache[text] = {"cond": cond, "pooled_output": pooled}
        else:
            self.cache.move_to_end(text)
        return (
            [[self.cache[text]["cond"], {"pooled_output": self.cache[text]["pooled_output"]}]],
        )
