import faiss
import time
import numpy as np
# import clip
import torch
from PIL import Image
from torchvision import datasets
from tqdm import tqdm
import os
import pdb
import json
# from clip_index_funcs import clip_index
import torch.nn.functional as F
from utils import get_image_paths,find_bounding_box, save_image_with_overlay, cut_image, mean_pooling, split_feature_by_token_length
import cv2
from torchvision import transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import CLIPProcessor, AutoTokenizer, CLIPModel, AutoModel
from torch.utils.data import Dataset, DataLoader
from functools import partial
from predictor import SamPredictor
import datetime
from collections import OrderedDict

class FaissIndexManager:
    def __init__(self, root:str, max_indices=10, expiration_time=3600):
        self.root = root
        self.index_dict = {}  # 存储所有task的index
        self.max_indices = max_indices
        self.expiration_time = expiration_time  # 1小时未访问的时间阈值
        self.access_time = OrderedDict()  # 记录最后访问时间

    def get_time(self):
        timestamp = time.time()
        cur_time = datetime.datetime.fromtimestamp(timestamp) # + datetime.timedelta(hours=8)
        cur_time_str = cur_time.strftime('%Y-%m-%d %H:%M:%S')
        return timestamp, cur_time_str

    def _load_faiss_index(self, user_id, task_id, modality, tag):
        # sam_demo_workdir
        # index = faiss.IndexFlatL2(128)
        index  = faiss.read_index(os.path.join(self.root, f"{user_id}/{task_id}/{modality}/faiss/{tag}_faiss.{modality}"))
        tag2dbid = np.load(os.path.join(self.root, f"{user_id}/{task_id}/{modality}/faiss/tag2dbid_{modality}.npy"), allow_pickle=True).item()
        return index, tag2dbid[tag]

    def _unload_least_recently_used(self):
        # OrderedDict 直接支持从左到右排序，删除第一个元素即可
        oldest_task_tag, _ = self.access_time.popitem(last=False)
        user_task_id, tag = oldest_task_tag
        if user_task_id in self.index_dict:
            del self.index_dict[user_task_id][tag]
            if not self.index_dict[user_task_id]:  # 如果user_task_id下已经没有任何tag了，删除user_task_id
                del self.index_dict[user_task_id]
        print(f"Unloaded {user_task_id} - {tag} faiss index.")
        print(self.access_time)

    def _remove_expired_indices(self):
        current_time = self.get_time()[0]
        expired_keys = [key for key, last_access in self.access_time.items()
                        if current_time - last_access[0] > self.expiration_time]
        for key in expired_keys:
            user_task_id, tag = key
            del self.access_time[key]
            if user_task_id in self.index_dict and tag in self.index_dict[user_task_id]:
                del self.index_dict[user_task_id][tag]
                if not self.index_dict[user_task_id]:  # 如果user_task_id下已经没有任何tag了，删除user_task_id
                    del self.index_dict[user_task_id]
            print(f"Automatically removed expired {user_task_id} - {tag} faiss index.")

    def get_faiss_index(self, user_id, task_id, modality, tag):
        # 检查并移除过期的索引
        self._remove_expired_indices()
        search_key = f"{user_id}_{task_id}: {modality}"
        if search_key in self.index_dict and tag in self.index_dict[search_key]:
            # 如果索引已经存在，更新访问时间和顺序
            self.access_time[(search_key, tag)] = self.get_time()
            self.access_time.move_to_end((search_key, tag))
            return self.index_dict[search_key][tag]
        else:
            # 如果索引不存在，则加载它
            if len(self.access_time) >= self.max_indices:
                self._unload_least_recently_used()

            if search_key not in self.index_dict:
                self.index_dict[search_key] = {}

            index = self._load_faiss_index(user_id, task_id, modality, tag)
            self.index_dict[search_key][tag] = index
            self.access_time[(search_key, tag)] = self.get_time()
            print(f"Loaded {search_key} - {tag} faiss index.")
            return index

# manager = FaissIndexManager(root="/mnt/A/", max_indices=3, expiration_time=3600)

# index1 = manager.get_faiss_index('user1', 'task1', 'tag1')
# index2 = manager.get_faiss_index('user1', 'task2', 'tag2')
# index3 = manager.get_faiss_index('user1', 'task1', 'tag3')
# index4 = manager.get_faiss_index('user2', 'task2', 'tag2')
# index5 = manager.get_faiss_index('user1', 'task2', 'tag3')
# index5 = manager.get_faiss_index('user1', 'task3', 'tag3')

# exit()

def find_knn_with_faiss(embeddings, embeddings2=None, k=10):

    faiss.normalize_L2(embeddings)
    if embeddings2 is None:
        embeddings2 = embeddings
    else:
        faiss.normalize_L2(embeddings2)

    # res = faiss.StandardGpuResources()
    # index_gpu = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(embeddings.shape[1]))
    #
    # # add data to index
    # index_gpu.add(embeddings2)
    #
    # # find k-nn
    # values, indices = index_gpu.search(embeddings, k)

    # 假设 embeddings 是你的特征矩阵
    d = embeddings2.shape[1]  # 向量维度

    # 创建一个 CPU 索引（IndexFlatIP 使用内积进行最近邻搜索）
    index_cpu = faiss.IndexFlatIP(d)

    # 添加特征到索引中
    index_cpu.add(embeddings2)

    # 查询特征
    query = embeddings[0:1]  # 示例查询，取第一条数据
    k = 5  # 取前 5 个最近邻
    values, indices = index_cpu.search(query, k)

    print("最近邻的索引:", indices)
    print("最近邻的距离:", values)

    return indices, values

def find_same_features(embeddings, embeddings2=None, thre=0.3, k=10):
    '''
        find according to embedding
        input: embeddings
        output: idx that are the same
    '''
    time0 = time.time()
    indices, values = find_knn_with_faiss(embeddings, embeddings2, k=k)
    print(f'search {k}-NN of embeddings of size {embeddings.shape} in {time.time()-time0} s')
    repeated_idx = []
    repeated_val = []
    _values = values.copy()
    values = values>thre


    for i, idx_i in enumerate(indices):
        _tmp = idx_i[values[i]].tolist()
        _tmp_val = _values[i][values[i]].tolist()

        # _tmp = [_tmp[0]] + _tmp
        # _tmp_val = [1] + _tmp_val
        # if len(_tmp) > 1:
        repeated_idx.append(_tmp)
        repeated_val.append(_tmp_val)
    return repeated_idx, repeated_val


class LabelWorker:
    def __init__(self, root:str):
        self.root = root
        self.faiss_manager = FaissIndexManager(root=root, max_indices=3, expiration_time=3600)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.msk2msk_thre = 0.85 # TODO: 0.6
        # self.img2img_thre = 0.7
        self.msk2msk_thre = 0.8
        self.img2img_thre = 0.7
        self.label_thre = 0.80 # TODO: 0.8

        # image mask
        sam_checkpoint=root+"sam_model/sam_vit_h_4b8939.pth"
        model_type="vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

        # image
        self.image_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        self.image_model.to(self.device)
        self.preprocess = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        self.num_results = 30
        self.batch_size = 16
        self.image_processor_with_tensors = partial(self.preprocess.image_processor, return_tensors="pt")

        # text
        model_name = 'BAAI/bge-m3'
        self.text_model = AutoModel.from_pretrained(model_name)
        self.text_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer = partial(tokenizer, padding=True, truncation=True, return_tensors='pt')
        self.sep_token = "<|SEP|>"
        self.txt2txt_thre = 0.2



    def clip_img_encoder(self, input):
        input['pixel_values'] = input['pixel_values'].squeeze(1)
        return self.image_model.get_image_features(**input).data


    def clip_pred_label(self, image, prompt=None, label_space=None):
        # image_path = "path_to_your_image.jpg"
        label_space = [prompt + i for i in label_space]
        probs = []
        # pdb.set_trace()
        for batch_idx_i in range(0, len(image), self.batch_size):
            inputs = self.preprocess(text=label_space, images=image[batch_idx_i*self.batch_size:(batch_idx_i+1)*self.batch_size], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.image_model(**inputs)
                # logits_per_image = outputs.logits_per_image.view(2, -1)  #
                logits_per_image = outputs.logits_per_image  #]

                # 取出最后一列
                last_column = logits_per_image[:, -1]
                # 将每一行的最后一个元素追加到这一行前面，使每一行有6个元素
                extended_tensor = torch.cat([logits_per_image[:, :-1]] + [last_column.unsqueeze(1)] * (logits_per_image.shape[1] - 1), dim=1)
                # 将每一行转换为 2x3 的矩阵
                result = extended_tensor.view(-1, 2, logits_per_image.shape[1] - 1)
                # pdb.set_trace()
                _probs = result.softmax(dim=1)
                probs.append(_probs)
        # pdb.set_trace()
        probs = torch.cat(probs)
        output = []
        for _prob in probs:
            print("-----------")
            output.append([label_space[i][len(prompt):] for i in range(len(_prob[0])) if _prob[0][i] > self.label_thre])
            for label, prob in zip(label_space, _prob[0]):
                print(f"{label}: {prob.item():.4f}")
        return output

    def load_image_with_dbid(self, user_id, task_id, sel_dbid):
        if "." in f"{sel_dbid}":
            pimage = os.path.join(self.root, f"{user_id}/image/{sel_dbid}")
            image = cv2.imread(pimage)
        else:
            pimages = [os.path.join(self.root, f"{user_id}/image/{sel_dbid}{suffix}") for suffix in ('.png', '.jpg', '.jpeg')]
            for pimage in pimages:
                image = cv2.imread(pimage)
                if image is not None:
                    break
        user_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # user_image = Image.open(pimage).convert('RGB')
        return user_image, pimage

    def get_emb_for_sam(self, user_id, task_id, sel_dbid):
        user_image, _ = self.load_image_with_dbid(user_id, task_id, sel_dbid)
        original_size, input_size, embeddings = self.sam_predictor.set_image(user_image)
        return embeddings.cpu().numpy()

    def predict_image(self, user_id, task_id, sel_dbid, tag, modality="image", prompt="Photo with ", label_space=None):

        if "others" in label_space:
            label_space.remove("others")
        label_space.append("others")

        faiss_index, db_id = self.faiss_manager.get_faiss_index(user_id, task_id, modality, tag)
        user_image, pimage = self.load_image_with_dbid(user_id, task_id, sel_dbid)
        # label_list = self.clip_pred_label(image=[user_image], prompt=prompt, label_space=label_space)
        similar_idx, similar_val = self.find_similar_image(faiss_index, user_image)
        result = {
            db_id[similar_idx[i]]:
            {
                tag: similar_val[i]
            }
            for i in range(len(similar_idx))
        }
        return result



    def predict_text(self, user_id, task_id, sel_dbid, tag, sel_text=None, modality="text", prompt="Photo with ", label_space=None):

        if "others" in label_space:
            label_space.remove("others")
        label_space.append("others")

        faiss_index, db_id = self.faiss_manager.get_faiss_index(user_id, task_id, modality, tag)
        if len(sel_text) > 0:
            user_text = sel_text
        else:
            raise RuntimeError(f"sel_text must be specified!")
        similar_idx, similar_val = self.find_similar_text(faiss_index, user_text)
        result = {
            db_id[similar_idx[i]]:
            {
                tag: similar_val[i]
            }
            for i in range(len(similar_idx))
        }
        return result


    def predict_sam(self, user_id, task_id, sel_dbid, tag, coordinate=[(23, 76, 1), (53, 86, 0)], prompt="Photo with ", label_space=None):

        if "others" in label_space:
            label_space.remove("others")
        label_space.append("others")

        faiss_index, db_id = self.faiss_manager.get_faiss_index(user_id, task_id,"image_mask", tag)
        user_image, pimage = self.load_image_with_dbid(user_id, task_id, sel_dbid)
        original_size, input_size, features = self.sam_predictor.set_image(user_image)
        input_point = np.array([[_co[0], _co[1]] for _co in coordinate])
        input_label = np.array([_co[2] for _co in coordinate])

        masks, scores, logits = self.sam_predictor.predict(point_coords=input_point,point_labels=input_label,multimask_output=True, original_size=original_size, input_size=input_size, features=features)
        select_mask = masks[np.argmax(scores)]
        # pdb.set_trace()
        # save_image_with_overlay(image_path=img_path, mask=select_mask, output_path= "./output/org" + img_path.split(".")[-2][1:] + f"_mask.jpeg", mask_color=(255, 0, 0), alpha=0.5, max_dimension=self.small_image_dim)
        cut_user_image = cut_image(user_image, select_mask)

        # image = Image.fromarray(cut_user_image.astype('uint8'))
        # # Save the image to a file
        # image.save('output_image.png')
        # label_list = self.clip_pred_label(image=[cut_user_image], prompt=prompt, label_space=label_space)

        # similar_idx, similar_val = self.find_similar_mask(faiss_index, cut_user_image, user_image)
        similar_idx, similar_val = self.find_similar_mask(faiss_index, cut_user_image)


        # similar_idx, similar_val = self.find_similar_mask(faiss_index, Image.open(pimage).convert('RGB'))
        result = {
            db_id[similar_idx[i]]:
            {
                tag: similar_val[i]
            }
            for i in range(len(similar_idx))
        }
        return result
        # pdb.set_trace()
        # _label_list = self.clip_pred_label(image=[cut_user_image], prompt=prompt, label_space=label_space)
        # label_list_out = []
        # img_np_out = []
        # idx_out = []
        # for i, _labels_i in enumerate(_label_list):
        #     valid_flag = False
        #     if len(_labels_i) > 0:
        #         for j, __labels_j in enumerate(_labels_i):
        #             if __labels_j in label_list[0]:
        #                 valid_flag = True
        #             else:
        #                 valid_flag = False
        #                 break
        #         if valid_flag:
        #             label_list_out.append(_labels_i)
        #             img_np_out.append(img_np[i])
        #             idx_out.append(i)
        # print(f"Valid index is {idx_out}")


        # return label_list, img_np_out

    def extract_embedding_batch(self, encoder, batch_feature, modality):
        if modality == "image":
            with torch.no_grad(), torch.cuda.amp.autocast():
                embedding = encoder(batch_feature.to(self.device))
                embedding_out = F.normalize(embedding, p=2, dim=1)
            return embedding_out.cpu().numpy().tolist()
        elif modality == "text":    
            encoder, tokenizer = encoder
            def token_st(sentences):
                return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            splitted_features = []
            lengths = []
            embedding_out_tmp = []
            # self.logger.info(batch_feature)
            for i, feature_i in enumerate(batch_feature):
                # print(f"length of feature_i: {len(feature_i)}")
                if tokenizer.sep_token is not None:
                    feature_i = feature_i.replace(self.sep_token, tokenizer.sep_token)
                _splitted_features = split_feature_by_token_length(feature_i, tokenizer, max_token_len=min(tokenizer.model_max_length, 2048))
                splitted_features.append(_splitted_features)
                if len(_splitted_features) == 0:
                    print(f"feature_i: {feature_i}")
                lengths.append(len(_splitted_features))
            flattened_features = [item for sublist in splitted_features for item in sublist]
            for i in range(0, len(flattened_features), self.batch_size):
                encoded_input = token_st(flattened_features[i:min(i+self.batch_size, len(flattened_features))]).to(self.device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    model_output = encoder(**encoded_input)
                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                # Normalize embeddings
                embedding = F.normalize(sentence_embeddings, p=2, dim=1)
                embedding_out_tmp.extend(embedding.cpu().numpy())

            start = 0
            embedding_out = []
            for length in lengths:
                a = embedding_out_tmp[start:start+length]
                if a == []:
                    print(f"start: {start}, length: {length}")
                    ValueError("Empty embedding")
                embedding_out.append(np.mean(a, 0).tolist())
                start += length
            return embedding_out

    def find_similar_image(self, faiss_index, user_image):
        '''
            Input: 
                - faiss_index for original image
                - user_iamge
        '''
        # find similar images
        image = self.image_processor_with_tensors([user_image])
        cut_user_img_emb = np.asarray(self.extract_embedding_batch(self.clip_img_encoder, image, modality="image"))

        d, i = faiss_index.search(cut_user_img_emb.reshape(1,-1), self.num_results)
        repeated_idx = [[]]
        repeated_val = [[]]
        i_img, d_img = i[0], d[0]
        sel_idx = set(i_img)

        for idx_i, ei in enumerate(i_img):
            if d_img[idx_i] > self.img2img_thre and ei in sel_idx:
                repeated_idx[-1].append(ei)
                repeated_val[-1].append(d_img[idx_i].astype(float))
        return repeated_idx[0], repeated_val[0]



    def find_similar_text(self, faiss_index, user_text):
        '''
            Input: 
                - faiss_index for original image
                - user_iamge
        '''
        # find similar images

        # inputs = self.tokenizer([user_text]).to(self.device)
        # with torch.no_grad():
        #     model_output = self.model(**inputs)
        # # Perform pooling
        # sentence_embeddings = mean_pooling(model_output, inputs['attention_mask'])
        # # Normalize embeddings
        # outputs = F.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()
        encoder = (self.text_model, self.tokenizer)
        cut_user_txt_emb = np.asarray(self.extract_embedding_batch(encoder, [user_text], modality="text"))

        d, i = faiss_index.search(cut_user_txt_emb.reshape(1,-1), self.num_results)
        repeated_idx = [[]]
        repeated_val = [[]]
        i_txt, d_txt = i[0], d[0]
        sel_idx = set(i_txt)

        for idx_i, ei in enumerate(i_txt):
            if d_txt[idx_i] > self.txt2txt_thre and ei in sel_idx:
                repeated_idx[-1].append(ei)
                repeated_val[-1].append(d_txt[idx_i].astype(float))
        return repeated_idx[0], repeated_val[0]


    def find_similar_mask(self, faiss_index, cut_user_image, user_image=None):

        # find similar images
        image = self.image_processor_with_tensors([cut_user_image])
        cut_user_img_emb = np.asarray(self.extract_embedding_batch(self.clip_img_encoder, image, modality="image"))

        d, i = faiss_index.search(cut_user_img_emb.reshape(1,-1), self.num_results)
        repeated_idx = [[]]
        repeated_val = [[]]
        i_img, d_img = i[0], d[0]
        if user_image is not None:
            image = self.image_processor_with_tensors([user_image])
            user_img_emb = np.asarray(self.extract_embedding_batch(self.clip_img_encoder, image, modality="image"))
            d, i = faiss_index.search(user_img_emb.reshape(1,-1), self.num_results)
            i_user_img, d_user_img = i[0], d[0]
            repeated_idx = [[]]
            repeated_val = [[]]
            for idx_i, ei in enumerate(i_user_img):
                if d_user_img[idx_i] > self.img2img_thre:
                    repeated_idx[-1].append(ei)
                    repeated_val[-1].append(d_user_img[idx_i].astype(float))
            sel_idx = set(repeated_idx[0])
        else:
            sel_idx = set(i_img)

        for idx_i, ei in enumerate(i_img):
            if d_img[idx_i] > self.msk2msk_thre and ei in sel_idx:
                repeated_idx[-1].append(ei)
                repeated_val[-1].append(d_img[idx_i].astype(float))
        return repeated_idx[0], repeated_val[0]
        # pdb.set_trace()
        # sel_masks = [(self.mask_img_name[i], self.masks_all[i]) for i in repeated_idx[0] if self.mask_img_name[i] in sel_imgs]
        # # return sel_masks
        # img_np = []
        # cut_img = []
        # for i, _mask in enumerate(sel_masks):
        #     _img_np, _cut_img = save_image_with_overlay(image_path=_mask[0], mask=_mask[1]['segmentation'], output_path= "./output/new_" + img_path.split(".")[-2][1:] + f"_mask_{i}.jpeg", mask_color=(255, 0, 0), alpha=0.5, max_dimension=self.small_image_dim)
        #     img_np.append(_img_np)
        #     cut_img.append(_cut_img)
        # return img_np, cut_img

if __name__ == "__main__":
    pass
    # label_space=["dog", "cat", "pig", "others"] # last one MUST be "others" or its substitutes
    # prompt = "photo with " 

    # faiss_idx = FaissIndex(force_update=True, prompt=prompt, label_space=label_space)
    faiss_idx = LabelWorker(root="/mnt/data1/zwzhu/", sam_checkpoint="/mnt/data1/zwzhu/A/dlp/sam_model/sam_vit_h_4b8939.pth", model_type="vit_h")
    # faiss_idx = LabelWorker(root="/mnt/A/", sam_checkpoint="/mnt/A/sam_model/sam_vit_h_4b8939.pth", model_type="vit_h")

    tiny_image_id_score = faiss_idx.predict_image(user_id="dataLabelTest", task_id="sam_demo_workdir", modality="image_mask", sel_dbid=1826899937808486473, label_space=["apple", "plum", "pear"] + ["others"], tag="apple", prompt="Photo with ")
    print(tiny_image_id_score)
    # # image_path = "./1.png"
    # # # # image_path = "/home/zwzhu/demo-SAM/img/key_images/1.jpeg"
    # # # image = Image.open(image_path)
    # # # faiss_idx.clip_pred_label(image)


    # # # init 拿emb
    # # # FE-> ML:
    # # # image path
    # # # ML -> FE:
    # # # emb
    # # "/mnt/server0-A/dataLabelTest/sam_demo_workdir/", 1824348320596889691
    # embeddings = faiss_idx.get_emb_for_sam(user_id="dataLabelTest", task_id="sam_demo_workdir", sel_dbid=1824348320596889691)
    # # pdb.set_trace()

    # # # FE -> ML:
    # # # image path
    # # # coordinates (x,y,type)[]
    # # # ML -> FE:
    # # # clip label
    # # # knn image 缩略图
    # # # display clip的返回值
    # # # 带mask的knn image缩略图display
    # # # 20, 13, 14, 16
    # # # (20, 16) --> (0.9, 0.1)
    # # # (13, 16) --> (0.1, 0.9)
    # # # (14, 16) --> (0.3, 0.7)


    # # # label_list, img_np = faiss_idx.predict_sam(image_path, coordinate=[(700, 350, 1), (53, 86, 0)], prompt=prompt, label_space=label_space)
    # tiny_image_id_score = faiss_idx.predict_sam(user_id="dataLabelTest", task_id="sam_demo_workdir", sel_dbid=1824348320596889691, coordinate=[(350, 350, 1)], label_space=["apple", "plum", "pear"] + ["others"], tag="apple", prompt="Photo with ")
    # # pdb.set_trace()
    # # print("ll", label_list)
    # # # print(img_np[0].shape)






