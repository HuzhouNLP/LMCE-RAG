from transformers import PretrainedConfig


class LKIFConfig(PretrainedConfig):
    def __init__(
        self,
        base_model_name_or_path: str = "",
        lkif_layer_frequency: int = 3,
        lkif_scale_factor: int | None = None,
        top_k_lkif: int = 100,
        dynamic_sparsify: bool = False,
        sep_query_head: bool = False,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.base_model_name_or_path = base_model_name_or_path
        # New LKIF-prefixed attributes
        self.lkif_layer_frequency = lkif_layer_frequency
        self.lkif_scale_factor = lkif_scale_factor
        self.top_k_lkif = top_k_lkif
        # Legacy KB-prefixed attributes for compatibility
        self.kb_layer_frequency = lkif_layer_frequency
        self.kb_scale_factor = lkif_scale_factor
        self.top_k_kb = top_k_lkif

        self.dynamic_sparsify = dynamic_sparsify
        self.sep_query_head = sep_query_head
        self.attn_implementation = attn_implementation
        super().__init__(**kwargs)
