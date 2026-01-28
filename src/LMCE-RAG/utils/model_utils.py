from transformers import AutoModelForCausalLM, AutoTokenizer

from lkif.models.kblam_processor import EncoderArgs, LKIFProcessor
from lkif.models.llama3_model import LkifLlamaForCausalLM
from lkif.models.phi3_model import LKIFPhi3ForCausalLM


def load_model_and_processor(
    model_path: str,
    tokenizer_path: str,
    encoder_name: str,
    kb_layer_frequency: int,
    encoder_dir: str,
    endpoint_url: str,
    token: str | None = None,
    get_oai_embd_online: bool = False,
) -> tuple[AutoModelForCausalLM, KBLaMProcessor]:
    if "llama" in model_path:
        model = LkifLlamaForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        ).bfloat16()
    else:
        model = LKIFPhi3ForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        ).bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    args = EncoderArgs(
        encoder_name=encoder_name,
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        kb_layer_frequency=kb_layer_frequency,
        encoder_dir=encoder_dir,
        projector_type="linear",
        endpoint_url=endpoint_url,
    )

    processor = LKIFProcessor(tokenizer, args, get_oai_embd_online, token)
    return model, processor
