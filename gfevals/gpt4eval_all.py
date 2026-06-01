from common import load_utils
from instructions.prompt_all import system_prompt

import os
import os.path as osp
import pandas as pd
import numpy as np
from collections import defaultdict
import logging as log
from tqdm import tqdm
from PIL import Image
import base64
from openai import OpenAI
from io import BytesIO
import rembg

os.environ["OPENAI_LOG"] = "none"
os.environ['OPENAI_API_KEY'] = ''

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='[%(asctime)s - %(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
log.getLogger("openai").setLevel(log.ERROR)
log.getLogger("httpx").setLevel(log.ERROR)
    
class GPT4Eval:
    def __init__(self, data_dir, pair_file_name, methods):
        self.data_dir = data_dir
        self.pair_file_name = pair_file_name
        self.methods = methods
        
        message = f"Loading model pairs from {osp.join(self.data_dir, 'files', self.pair_file_name)}"
        log.info(message)
        self.model_pairs = load_utils.load_json(osp.join(self.data_dir, 'files', self.pair_file_name))

        self.out_dir = osp.join(self.data_dir, 'results')
        self.app_dataset, self.struct_dataset, self.cat_type = self.pair_file_name.split('_')[:3]

        self.app_dataset = self.app_dataset.upper() + '_data'
        self.struct_dataset = self.struct_dataset.upper() + '_data'
        self.simple_as_struct = self.struct_dataset == 'SP_data'

        self.app_data_dir = osp.join(self.data_dir, self.app_dataset)
        self.struct_data_dir = osp.join(self.data_dir, self.struct_dataset)

        print(f"Appearance dataset: {self.app_dataset}, Structure dataset: {self.struct_dataset}, Cat type: {self.cat_type}")
        
        self.client = OpenAI()
        self.metric_names = [
            "Style Fidelity",
            "Structure Clarity",
            "Style Integration",
            "Detail Quality",
            "Shape Adaptation",
            "Overall Quality"
        ]
        self.scores = defaultdict(lambda: defaultdict(list))
        self.rembg_session = None
    
    def encode_and_resize_img(self, img_path):
        image = Image.open(img_path)
        image = image.convert('RGB')  # Ensure image is in RGB format
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # or "JPEG" depending on use case
        img_bytes = buffered.getvalue()
        
        image = base64.b64encode(img_bytes).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image}",
                "detail": "high"
            }
        }
    
    def run(self): 
        self.model_pairs = self.model_pairs
        for model_pair in tqdm(self.model_pairs, total=len(self.model_pairs), desc=f'Processing {self.__class__.__name__}'):
            self.run_per_pair(model_pair)
        
        log.info(f"Finished processing {self.__class__.__name__}")
        # print(self.scores)
        print("\n==== Mean Rankings Per Method ====\n")
        data = []
        for method in self.methods:
            row = {'Method': method}
            for metric in self.metric_names:
                scores = self.scores[method][metric]
                row[metric] = np.mean(scores) if scores else None
            data.append(row)

        df = pd.DataFrame(data)
        df = df.set_index("Method")
        print(df)
        
        # Convert to a regular dictionary
        def convert_defaultdict(d):
            if isinstance(d, defaultdict):
                d = {k: convert_defaultdict(v) for k, v in d.items()}
            return d

        # regular_scores = convert_defaultdict(self.scores)
        # np.save('scores.npy', regular_scores)
    
    def run_per_pair(self, model_pair):
        app_model_info = model_pair[0]
        struct_model_info = model_pair[1]
        
        app_model_name = app_model_info['model_name']   
        struct_model_name = struct_model_info['model_name'] 
        
        pair_out_dir = osp.join(self.out_dir, self.pair_file_name[:-5], app_model_name +'_' + struct_model_name)
        
        view_filenames = ['out_meshview_1.png', 'out_meshview_2.png']
        
        for view_filename in view_filenames:
            # Appearance Image
            app_image_path = osp.join(self.app_data_dir, 'models', app_model_name, app_model_info['img_name'])
            if not osp.exists(app_image_path):
                log.warning(f"Missing appearance image: {app_image_path}")
                continue
            app_image = self.encode_and_resize_img(app_image_path)

            # Structure Image
            struct_image_path = osp.join(
                self.struct_data_dir,
                'processed' if self.simple_as_struct else 'models',
                struct_model_name,
                view_filename
            )
            if not osp.exists(struct_image_path):
                log.warning(f"Missing structure image: {struct_image_path}")
                continue
            struct_image = self.encode_and_resize_img(struct_image_path)

            # Output Images
            method_to_image = {}
            for method in self.methods:
                img_path = osp.join(pair_out_dir, method, view_filename)
                if osp.exists(img_path):
                    image = self.encode_and_resize_img(img_path)
                    method_to_image[method] = image
                else:
                    log.warning(f"Missing image for method {method} at view {view_filename}")

            if len(method_to_image) < len(self.methods):
                log.warning(f"Skipping view {view_filename}: not all method images found.")
                continue

            # Shuffle + map to A/B/C/D/E/F
            items = list(method_to_image.items())
            label_map = ['A', 'B', 'C', 'D', 'E', 'F']
            images = {label: image for label, (_, image) in zip(label_map, items)}
            reverse_map = {label: method for label, (method, _) in zip(label_map, items)}
            
            assert len(label_map) == len(self.methods), f"Expected {len(self.methods)} methods, but got {len(label_map)} labels."
            
            evaluation_result = self.client.chat.completions.create(
                model='gpt-5-mini',
                # temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Structure Image:"
                            },
                            struct_image,
                            {
                                "type": "text",
                                "text": "Style Image:"
                            },
                            app_image,
                            {
                                "type": "text",
                                "text": "Output A:"
                            },
                            images['A'],
                            {
                                "type": "text",
                                "text": "Output B:"
                            },
                            images['B'],
                            {
                                "type": "text",
                                "text": "Output C:"
                            },
                            images['C'],
                            {
                                "type": "text",
                                "text": "Output D:"
                            },
                            images['D'],
                            {
                                "type": "text",
                                "text": "Output E:"
                            },
                            images['E'],
                            {
                                "type": "text",
                                "text": "Output F:"
                            },
                            images['F'],
                            {
                                "type": "text",
                                "text": "Please judge each output independently, irrespective of the order in which they are shown. Evaluate all the images in the same way irrespective of order and return a single string in the format specified above, ranking the outputs from best (1) to worst (6) for each of the six criteria. Use ties where appropriate."
                            }
                        ]
                    }
                ],
                n=3
            )
            
            # Parse GPT responses
            for choice in evaluation_result.choices:
                content = choice.message.content.strip()

                # Handle with or without "Final answer:"
                if "Final answer:" in content:
                    answer_str = content.split("Final answer:")[1].strip()
                else:
                    fallback_str = content.strip()
                    if fallback_str.count("/") == 5 and all(len(part.strip().split()) == 3 for part in fallback_str.split("/")):
                        answer_str = fallback_str
                        log.warning("Missing 'Final answer:' prefix, but accepted based on format.")
                    else:
                        log.warning(f"Missing 'Final answer:' and bad format: {content}")
                        continue

                metric_rankings = [m.strip() for m in answer_str.split("/")]

                if len(metric_rankings) != 6:
                    log.warning(f"Unexpected number of metrics: {metric_rankings}")
                    continue

                for metric_idx, rank_str in enumerate(metric_rankings):
                    try:
                        ranks = list(map(int, rank_str.split()))
                        for label, rank in zip(label_map, ranks):
                            method = reverse_map[label]
                            self.scores[method][self.metric_names[metric_idx]].append(rank)
                    except Exception as e:
                        log.warning(f"Failed to parse or map ranking '{rank_str}': {e}")

if __name__ == '__main__':
    data_dir = '/drive/dumps/shape-rep/dataset'
    # pair_filenames = ['abo_sp_same_cat_pairs.json', 'abo_sp_diff_cat_pairs.json', 'abo_abo_diff_cat_pairs.json', 'abo_abo_same_cat_pairs.json']
    pair_filenames = ['abo_abo_diff_cat_pairs.json', 'abo_abo_same_cat_pairs.json']

    for pair_file_name in pair_filenames:
        methods = ['opt_img', 'trellis_detail_variation_image', 'easitex', 'cross-image-attention', 'mambast', 'vertex_nn_mesh']
        gpt4eval = GPT4Eval(data_dir, pair_file_name, methods)
        gpt4eval.run()
