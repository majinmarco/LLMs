import streamlit as st
import torch
from transformers import AutoTokenizer

from peft import AutoPeftModelForCausalLM


peft_repo = "marconardone/Mistral-7B-Trump-Edition"
base_repo = "mistralai/Mistral-7B-v0.1"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     #bnb_4bit_compute_dtype=torch.bfloat16
# ) # loads in bnb config from last time

# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_id,  # Mistral, same as before
#     quantization_config=bnb_config,  # Same quantization config as before
#     device_map="auto",
#     trust_remote_code=True,
# ) #loads in base model

fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(peft_repo)
fine_tuned_model = fine_tuned_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_repo, add_bos_token=True, trust_remote_code=True)

generation_config = fine_tuned_model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.repetition_penalty = 1.3
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config.do_sample = True
generation_config.include_prompt_in_result = False

def LLM_generator(topic):
  model_input = tokenizer(f"The following is a speech by Donald Trump about {topic}: # ", return_tensors = "pt").to("cuda")

  fine_tuned_model.eval()
  with torch.no_grad():
    outputs = fine_tuned_model.generate(
        input_ids = model_input.input_ids,
        attention_mask = model_input.attention_mask,
        generation_config = generation_config,
    )

  return tokenizer.decode(outputs[0], skip_special_tokens=True)+"[...]"

def main():
    st.set_page_config(page_title = "Trump Speech Generator", page_icon =":bird:")

    st.header("Trump Speech Generator")
    topic = st.text_area("Topic for speech:")

    if topic:
        st.write("Generating speech...")
        speech = LLM_generator(topic)
        st.info(speech)

if __name__ == "__main__":
    main()
  