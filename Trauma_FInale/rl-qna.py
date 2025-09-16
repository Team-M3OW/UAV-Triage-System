from transformers import AutoModel

model_path = "Efficient-Large-Model/LongVILA-R1-7B"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
model.config.llm_cfg['use_bfloat16'] = True
model.config.llm_cfg['max_length'] = 512
model.config.llm_cfg['min_length'] = 10
model.config.llm_cfg['temperature'] = 0.1
# You can adjust the FPS value as needed. 
# To disable FPS control, set it to 0 and manually specify the number of processed video frames via `num_video_frames`.
# Example:
# model.config.fps = 8.0
model.config.num_video_frames, model.config.fps = 8, 0


use_thinking = False # Switching between thinking and non-thinking modes
system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

prompt = "Can you see the face of the casualty ? if yes, classify the casualty on the basis of the skin texture, otherwise reply 'face not visible'"
video_path = "/home/uasdtu/Documents/Chirag/Trauma-Darpa/casualty_videos/P2D2_G1_S12.MP4"

if use_thinking:
  prompt = system_prompt_thinking.format(question=prompt)

response = model.generate_content([prompt, {"path": video_path}])
print("Response: ", response)
