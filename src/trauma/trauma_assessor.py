import re
import time
import pandas as pd
import torch
from tqdm import tqdm


def check_tags(s):
    pattern = r'<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>'
    match = re.search(pattern, s)
    return bool(match)


def parse_to_dict(s):
    s = s.strip().strip('{}')
    items = s.split(',')
    result = {}
    for item in items:
        if ':' in item:
            key, value = item.split(':', 1)
            result[key.strip()] = value.strip()
    return result


class TraumaAssessor:
    def __init__(self, model_path="Efficient-Large-Model/LongVILA-R1-7B", prompts_module=None):
        from transformers import AutoModel, GenerationConfig
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        self.GenerationConfig = GenerationConfig
        self.prompts_module = prompts_module

    def generate_report(self, video_path, extract_frames_fn):
        generation_config = self.model.default_generation_config
        gen_config = self.GenerationConfig(
            max_new_tokens=1028,
            max_length=generation_config.max_length,
            pad_token_id=generation_config.pad_token_id,
            bos_token_id=generation_config.bos_token_id,
            eos_token_id=generation_config.eos_token_id,
        )
        self.model.config.num_video_frames, self.model.config.fps = 8, 0
        self.model.config.llm_cfg['use_bfloat16'] = True
        self.model.config.llm_cfg['max_length'] = 512
        self.model.config.llm_cfg['min_length'] = 10
        self.model.config.llm_cfg['temperature'] = 0.1

        prompts = self.prompts_module()
        system_prompt_thinking = prompts.thinking()
        description = prompts.description()
        Report = prompts.report()
        Regen = prompts.regen()
        Regen_sh = prompts.regen_sh()
        pattern = prompts.regex()
        verify = prompts.verify()
        resp_sever = prompts.repiratory_severe()

        start = time.time()

        self.model.config.num_video_frames, self.model.config.fps = 8, 0
        frames = extract_frames_fn(video_path, num_frames=8)

        gen_config.max_new_tokens = 512
        desc_prompt = system_prompt_thinking.format(question=description)
        response_desc = self.model.generate_content([desc_prompt, frames], generation_config=gen_config)
        print(f"DESCRIPTION: {response_desc}")

        if not check_tags(response_desc):
            self.model.config.llm_cfg['temperature'] = 0.3
            self.model.config.num_video_frames, self.model.config.fps = 8, 0
            desc_prompt = system_prompt_thinking.format(question=description)
            response_desc = self.model.generate_content([desc_prompt, frames], generation_config=gen_config)
            print(f"DESCRIPTION (retry): {response_desc}")

        self.model.config.num_video_frames, self.model.config.fps = 8, 0
        gen_config.max_new_tokens = 256
        response_report = self.model.generate_content(
            ["Video Description:" + response_desc + Report, frames], generation_config=gen_config
        )
        print(f"\nReport: {response_report}")

        self.model.config.num_video_frames, self.model.config.fps = 8, 0
        if not bool(pattern.match(response_report.replace("\n", ""))):
            response_report = self.model.generate_content(
                ["Previous Response:" + response_report + Regen, frames], generation_config=gen_config
            )
        print(f"\nVerified Report: {response_report}")

        response_report_2 = parse_to_dict(str(response_report))
        if response_report_2.get("Upper Extremity") == "Wound" or response_report_2.get("Lower Extremity") == "Wound":
            self.model.config.num_video_frames, self.model.config.fps = 8, 0
            response_report = self.model.generate_content(
                [response_report + verify, frames], generation_config=gen_config
            )
            print(f"\nAmputation Verified: {response_report}")

        self.model.config.num_video_frames, self.model.config.fps = 8, 0
        if not bool(pattern.match(response_report.replace("\n", ""))):
            response_report = self.model.generate_content(
                ["Previous Response:" + response_report + Regen, frames], generation_config=gen_config
            )

        self.model.config.num_video_frames, self.model.config.fps = 8, 0
        response_report_sh = self.model.generate_content([resp_sever, frames], generation_config=gen_config)
        print(f"\nRespiratory/Hemorrhage: {response_report_sh}")

        self.model.config.num_video_frames, self.model.config.fps = 8, 0
        if not bool(pattern.match(response_report_sh.replace("\n", ""))):
            response_report_sh = self.model.generate_content(
                ["Previous Response:" + response_report_sh + Regen_sh, frames], generation_config=gen_config
            )

        new_df = {
            'video': [video_path],
            'response_trauma': [response_report],
            'thinking': [response_desc],
            'response_respiratory_severe': [response_report_sh],
        }
        new_df = pd.DataFrame(new_df)
        new_df.to_csv('final_desc_cropped.csv', mode='a', index=False, header=False)

        end = time.time()
        print(f"Time taken: {end - start:.2f}s")
        return response_report, response_desc, response_report_sh
