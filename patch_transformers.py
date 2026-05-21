"""
patch_transformers.py
=====================
Monkey-patches the Hugging Face `transformers` library to prevent the infamous
AttributeError: 'list' object has no attribute 'keys' when loading DeBERTa tokenizers.
This happens on newer versions of the `transformers` library because some cached or
fine-tuned models store `extra_special_tokens` as a list rather than a dictionary.
"""

import logging

logger = logging.getLogger(__name__)

def apply_patch():
    try:
        import transformers.tokenization_utils_base as base
        
        # Patch __init__
        orig_init = base.PreTrainedTokenizerBase.__init__
        def patched_init(self, *args, **kwargs):
            if "extra_special_tokens" in kwargs:
                est = kwargs["extra_special_tokens"]
                if isinstance(est, list) or not hasattr(est, "keys"):
                    kwargs["extra_special_tokens"] = {}
            orig_init(self, *args, **kwargs)
        base.PreTrainedTokenizerBase.__init__ = patched_init
        
        # Patch _set_model_specific_special_tokens
        orig_set_tokens = base.PreTrainedTokenizerBase._set_model_specific_special_tokens
        def patched_set_tokens(self, special_tokens):
            if isinstance(special_tokens, list) or not hasattr(special_tokens, "keys"):
                special_tokens = {}
            orig_set_tokens(self, special_tokens)
        base.PreTrainedTokenizerBase._set_model_specific_special_tokens = patched_set_tokens
        
        logger.debug("Successfully monkey-patched transformers PreTrainedTokenizerBase to prevent extra_special_tokens bug.")
    except Exception as e:
        logger.warning(f"Could not monkey-patch transformers PreTrainedTokenizerBase: {e}")

# Automatically apply the patch on import
apply_patch()
