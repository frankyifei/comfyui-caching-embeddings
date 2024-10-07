from .nodes import CachingCLIPTextEncode

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeWithCache": CLIPTextEncodeWithCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeWithCache": "CLIP Text Encode with Caching",
}
