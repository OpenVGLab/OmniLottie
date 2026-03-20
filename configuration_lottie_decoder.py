"""
OmniLottie Decoder Configuration
"""
from transformers import PretrainedConfig
from typing import Optional


class LottieDecoderConfig(PretrainedConfig):
    """
    Configuration class for LottieDecoder model, inheriting from PretrainedConfig

    Stores configuration parameters for the LottieDecoder model,
    supporting Hugging Face's standard save and load mechanisms.

    Args:
        pix_len (int): Maximum length for image/video tokens, default 4560
        text_len (int): Maximum length for text tokens, default 1500
        base_model_path (str): Path or name of base Qwen2.5-VL model
        vocab_size (int): Vocabulary size, extended to 192400 to support Lottie tokens
        bos_token_id (int): Beginning-of-sequence token ID for Lottie
        eos_token_id (int): End-of-sequence token ID for Lottie
        pad_token_id (int): Padding token ID
        torch_dtype (str): Model weight data type, default "bfloat16"
        attn_implementation (str): Attention implementation method, default "eager"
    """

    model_type = "lottie_decoder"

    def __init__(
        self,
        pix_len: int = 4560,
        text_len: int = 1500,
        base_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        vocab_size: int = 192400,
        bos_token_id: int = 192398,
        eos_token_id: int = 192399,
        pad_token_id: int = 151643,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs
        )

        self.pix_len = pix_len
        self.text_len = text_len
        self.base_model_path = base_model_path
        self.vocab_size = vocab_size
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
