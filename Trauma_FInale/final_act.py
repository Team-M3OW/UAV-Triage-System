#MK2

from transformers import AutoModel, GenerationConfig
from prompts import prompts
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import os
import re
from torchvision import transforms
import time
from PIL import Image
from video_to_cropped import extract_cropped_human_frames, extract_face_human_frames


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


def generate_report(video_path):
    generation_config = model.default_generation_config
    generation_config2 = GenerationConfig(
        max_new_tokens = 1028,
        max_length = generation_config.max_length,
        pad_token_id = generation_config.pad_token_id,
        bos_token_id = generation_config.bos_token_id,
        eos_token_id = generation_config.eos_token_id
    )
    model.config.num_video_frames, model.config.fps = 8, 0
    model.config.llm_cfg['use_bfloat16'] = True
    model.config.llm_cfg['max_length'] = 512
    model.config.llm_cfg['min_length'] = 10
    model.config.llm_cfg['temperature'] = 0.1

    # video_dir="/home/uasdtu/Documents/Chirag/Trauma-Darpa/casualty_videos"
    prompt_ = prompts()
    system_prompt_thinking = prompt_.thinking()
    description = prompt_.description()
    Report = prompt_.report()
    Regen = prompt_.regen()
    Regen_sh = prompt_.regen_sh()
    pattern = prompt_.regex()
    verify = prompt_.verify()
    resp_sever = prompt_.repiratory_severe()
    use_thinking = True 
    start=time.time()
    #####DESCRIPTION PROMPT#####
    model.config.num_video_frames, model.config.fps = 8, 0
    frames = extract_cropped_human_frames(video_path, num_frames=8)

    print(frames[0].size)
    print("Processing video: \n", video_path)
    use_thinking = True
    generation_config2.max_new_tokens = 512
    if use_thinking:
        description = system_prompt_thinking.format(question=description)
    response_desc = model.generate_content([description,frames], generation_config=generation_config2 ) 
    print(f"DESCRIPTION : {response_desc} \n")

    #####ORDER TAGS CHECK########
    if check_tags(response_desc)==False:
        print("MODEL IN THE LOOP")
        model.config.llm_cfg['temperature'] = 0.3    
        model.config.num_video_frames, model.config.fps = 8, 0
        print("Processing video: \n", video_path)
        use_thinking = True
        generation_config2.max_new_tokens = 512
        if use_thinking:
            description = system_prompt_thinking.format(question=description)
        response_desc = model.generate_content([description,frames], generation_config=generation_config2 ) 
        print(f"DESCRIPTION : {response_desc} \n")
    
    #####REPORT PROMPT#####
    model.config.num_video_frames, model.config.fps = 8, 0
    use_thinking = False
    generation_config2.max_new_tokens = 256
    if use_thinking:
        description = system_prompt_thinking.format(question=Report)
    response_report = model.generate_content(["Video Description:" + response_desc + Report,frames], generation_config=generation_config2)
    print(f"\nResponse: {response_report}")


    ####JSON VERIFY PROMPT###
    model.config.num_video_frames, model.config.fps = 8, 0
    use_thinking = False
    generation_config2.max_new_tokens = 256
    if bool(pattern.match(response_report.replace("\n", ""))):
        print("Valid format")
    else:
        response_report = model.generate_content(["Previous Response:"+ response_report + Regen,frames], generation_config=generation_config2)
    print(f"\nNew Response: {response_report}")
    print(type(response_report))

    ###VERIFY WITH RESPECT TO AMPUTATION###
    response_report_2 = parse_to_dict(str(response_report))
    if((response_report_2["Upper Extremity"] == "Wound") or (response_report_2["Lower Extremity"]=="Wound")):
        model.config.num_video_frames, model.config.fps = 8, 0
        use_thinking = False
        response_report = model.generate_content([response_report + verify, frames ], generation_config=generation_config2)
        print(f"\nVerified Response: {response_report}")

    ####JSON VERIFY PROMPT###
    model.config.num_video_frames, model.config.fps = 8, 0
    use_thinking = False
    if bool(pattern.match(response_report.replace("\n", ""))):
        print("Valid format")
    else:
        response_report = model.generate_content(["Previous Response:"+ response_report + Regen,frames], generation_config=generation_config2 )
    print(f"\nNew Response: {response_report}")

    ###RESPIRATORY AND SEVERE HEMORRHAGE REPORT###
    model.config.num_video_frames, model.config.fps = 8, 0
    use_thinking = False
    response_report_sh = model.generate_content([resp_sever, frames ], generation_config=generation_config2)
    print(f"\nRespiratory and Severe Hemorrhage Response: {response_report_sh}")

    ####JSON VERIFY PROMPT###
    model.config.num_video_frames, model.config.fps = 8, 0
    use_thinking = False
    if bool(pattern.match(response_report_sh.replace("\n", ""))):
        print("Valid format")
    else:
        response_report_sh = model.generate_content(["Previous Response:"+ response_report_sh + Regen_sh,frames], generation_config=generation_config2 )
    print(f"\nNew Response: {response_report_sh}")

    new_df={'video':[video_path],'response_trauma':[response_report],'thinking:':[response_desc], 'response_respiratory_severe':[response_report_sh]}
    new_df=pd.DataFrame(new_df)
    new_df.to_csv('./final_desc_cropped.csv',mode='a',index=False,header=False)
    end= time.time()
    print(f"Time taken Completely: {end-start}")

if __name__ == "__main__":
    model_path = "Efficient-Large-Model/LongVILA-R1-7B"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    generate_report("/home/uasdtu/Documents/Chirag/Trauma-Darpa/casualty_videos")
