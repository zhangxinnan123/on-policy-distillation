from vllm import LLM, SamplingParams

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # adjust as needed

llm = LLM(model="Qwen/Qwen3-8B", tensor_parallel_size=2, max_model_len=8192)
tokenizer = llm.get_tokenizer()

prompt_text = (
    "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. "
    "When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, "
    "including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, "
    "the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. "
    "Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the "
    "walk takes her, including the $t$ minutes spent in the coffee shop. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)
# Evaluate the integral \( \int_S y \, ds \) where \( S \) is the region in the plane \( z = 1 + y \) that lies inside the cone \( z = \sqrt{2(x^2 + y^2)} \). Determine the bounds of integration and compute the integral.
messages = [{"role": "user", "content": prompt_text}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

params = SamplingParams(logprobs=5)  # top 5 logprobs per token
outputs = llm.generate([prompt], params)

output = outputs[0].outputs[0]
logprobs_list = output.logprobs  # list of dicts, one per generated token
generated_token_ids = output.token_ids

print("Prompt:")
print(prompt)   

print(f"Generated {len(generated_token_ids)} tokens\n")
print("=" * 80)

for step, (token_id, top5) in enumerate(zip(generated_token_ids[:10], logprobs_list[:10])):
    chosen_token = tokenizer.decode([token_id])
    print(f"Step {step:4d} | Chosen: {repr(chosen_token):20s} (id={token_id})")
    for rank, (tid, logprob_obj) in enumerate(
        sorted(top5.items(), key=lambda x: x[1].logprob, reverse=True)
    ):
        tok = tokenizer.decode([tid])
        marker = " <-- chosen" if tid == token_id else ""
        print(f"         #{rank+1}: {repr(tok):20s} (id={tid:6d})  logprob={logprob_obj.logprob:.4f}{marker}")
    print()
