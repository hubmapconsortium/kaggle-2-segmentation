import copy
import gc
import json
import os
import random
import sys
import timeit
from argparse import ArgumentParser
from os import listdir, makedirs, path

import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import rasterio
import timm
import torch
import torch.nn.functional as F
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from coat import *
from rasterio.windows import Window
from skimage import measure
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
cudnn.benchmark = True
##Load config json
file = open('config.json')
data = json.load(file)
file.close()

config = data["predict"]
test_df = pd.read_csv(config["test_df"])

# data_dir = '.'
data_dir = config['data_dir']
output_dir = config['output_dir']
# data_dir = '../input/hubmap-organ-segmentation'
models_folder = config['models_folder']
# models_folder1 = config['models_folder1']
# models_folder2 = config['models_folder2']
# models_folder = '../input/subweights0/'

# df = pd.read_csv(path.join(data_dir, 'test.csv'))

supported_tissue_types = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']

class HuBMAPDataset(Dataset):
    def __init__(self, im_path, config,organ,new_size):
        super().__init__()
        self.im_path = im_path
        self.data = rasterio.open(self.im_path)
        self.organ = organ
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i,subdataset in enumerate(subdatasets,0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width # height and width of the slide (no resize)
        self.input_sz = new_size # input image size for U-net
        self.sz = config['resolution'] # tile size (no resize)
        # add to each input tile # Trick to avoid the edge effect and this pad size determines the size of the neglected part 
        # # (see the self.pred_sz below).
        self.pad_sz = config['pad_size'] 
        # self.pred_sz is the size used for prediction which is cut from the output (with self.sz) of U-net
        # For example, 
        # self.sz = 1024, self.pad_sz = 256, self.input_sz = 320
        # then I first resize 1024x1024 tile into 320x320 and feed it into U-net.
        # The output of U-net is 320x320 and I resize it to 1024x1024.
        # Since the self.pad_sz=256 here, I extract the center part 512x512 (512=1024-2*256) from the 1024x1024.
        self.pred_sz = self.sz - 2*self.pad_sz
        # pad size for the slide
        # Since the prediction size is self.pred_sz (not self.sz) here, I used the equation below
        self.pad_h = self.pred_sz - self.h % self.pred_sz # add to whole slide
        self.pad_w = self.pred_sz - self.w % self.pred_sz # add to whole slide
        # number of tiles 
        self.num_h = (self.h + self.pad_h) // self.pred_sz
        self.num_w = (self.w + self.pad_w) // self.pred_sz
        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx):
        # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h*self.pred_sz 
        x = i_w*self.pred_sz
        py0,py1 = max(0,y), min(y+self.pred_sz, self.h)
        px0,px1 = max(0,x), min(x+self.pred_sz, self.w)
        
        # padding coordinate for rasterio
        qy0,qy1 = max(0,y-self.pad_sz), min(y+self.pred_sz+self.pad_sz, self.h)
        qx0,qx1 = max(0,x-self.pad_sz), min(x+self.pred_sz+self.pad_sz, self.w)
        
        # placeholder for input tile (before resize)
        img = np.zeros((self.sz,self.sz,3), np.uint8)
        
        # replace the value
        if self.data.count == 3: 
            img[0:qy1-qy0, 0:qx1-qx0] =\
                np.moveaxis(self.data.read([1,2,3], window=Window.from_slices((qy0,qy1),(qx0,qx1))), 0,-1)
        else:
            for i,layer in enumerate(self.layers):
                img[0:qy1-qy0, 0:qx1-qx0, i] =\
                    layer.read(1,window=Window.from_slices((qy0,qy1),(qx0,qx1)))

        sample = {'id': self.im_path.split("/")[-1].split('.')[:-1], 
                  'organ': self.organ, 
                  'orig_h': self.sz, 
                  'orig_w': self.sz,
                  "p":[py0,py1,px0,px1],
                  "q":[qy0,qy1,qx0,qx1]}

        if self.sz != self.input_sz:

            img = cv2.resize(img, self.input_sz, interpolation=cv2.INTER_AREA)

        img = preprocess_inputs(img)
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()

        i=0
        sample['img{}'.format(i)] = img
        """""
        if self.sz != self.input_sz:
            img = cv2.resize(img, (self.input_sz, self.input_sz), interpolation=cv2.INTER_AREA)
        return {'img':img, 'p':[py0,py1,px0,px1], 'q':[qy0,qy1,qx0,qx1]}
        """
        return sample

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x

class TestDataset(Dataset):
    def __init__(self, df, data_dir='test_images', new_size=None):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.new_size = new_size

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        img0 = cv2.imread(path.join(self.data_dir, '{}.tiff'.format(r['id'])), cv2.IMREAD_UNCHANGED)

        orig_shape = img0.shape

        sample = {'id': r['id'], 'organ': r['organ'], 'data_source': r['data_source'], 'orig_h': orig_shape[0], 'orig_w': orig_shape[1]}

        for i in range(len(self.new_size)):

            img = cv2.resize(img0, self.new_size[i])

            img = preprocess_inputs(img)
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()

            sample['img{}'.format(i)] = img

        return sample
    
def my_collate_fn(batch):
    img = []
    p = []
    q = []
    id=[]
    orig_h=[]
    orig_w=[]
    organ=[]
    for sample in batch:
        img.append(sample['img0'])
        p.append(sample['p'])
        q.append(sample['q'])
        id.append(sample['id'])
        orig_h.append(sample['orig_h'])
        orig_w.append(sample['orig_w'])
        organ.append(sample['organ'])

    img = torch.stack(img)
    return {'img':img, 'p':p, 'q':q,'id':id,'orig_h':orig_h,'orig_w':orig_w,'organ':organ}


class ConvSilu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvSilu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class Timm_Unet(nn.Module):
    def __init__(self, name='resnet34', pretrained=True, inp_size=3, otp_size=1, decoder_filters=[32, 48, 64, 96, 128], **kwargs):
        super(Timm_Unet, self).__init__()

        if name.startswith('coat'):
            encoder = coat_lite_medium()

            if pretrained:
                checkpoint = config['model_ckpt']
                checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
                state_dict = checkpoint['model']
                encoder.load_state_dict(state_dict,strict=False)
        
            encoder_filters = encoder.embed_dims
        else:
            encoder = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=inp_size)

            encoder_filters = [f['num_chs'] for f in encoder.feature_info]

        decoder_filters = decoder_filters

        self.conv6 = ConvSilu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvSilu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvSilu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvSilu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvSilu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvSilu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvSilu(decoder_filters[-3], decoder_filters[-4])

        if len(encoder_filters) == 4:
            self.conv9_2 = None
        else:
            self.conv9_2 = ConvSilu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        
        self.conv10 = ConvSilu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], otp_size, 1, stride=1, padding=0)

        self.cls =  nn.Linear(encoder_filters[-1] * 2, 5)
        self.pix_sz =  nn.Linear(encoder_filters[-1] * 2, 1)

        self._initialize_weights()

        self.encoder = encoder

    def forward(self, x):
        batch_size, C, H, W = x.shape

        if self.conv9_2 is None:
            enc2, enc3, enc4, enc5 = self.encoder(x)
        else:
            enc1, enc2, enc3, enc4, enc5 = self.encoder(x)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))

        if self.conv9_2 is not None:
            dec9 = self.conv9_2(torch.cat([dec9, 
                    enc1
                    ], 1))
        
        dec10 = self.conv10(dec9) # F.interpolate(dec9, scale_factor=2))

        x1 = torch.cat([F.adaptive_avg_pool2d(enc5, output_size=1).view(batch_size, -1), 
                        F.adaptive_max_pool2d(enc5, output_size=1).view(batch_size, -1)], 1)

        # x1 = F.dropout(x1, p=0.3, training=self.training)
        organ_cls = self.cls(x1)
        pixel_size = self.pix_sz(x1)

        return self.res(dec10), organ_cls, pixel_size

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

test_batch_size = config['test_batch_size']

# amp_autocast = suppress
amp_autocast = torch.cuda.amp.autocast

# half_size = True

hubmap_only = False #True #False

organ_threshold = {
    'Hubmap': {
        'kidney'        : 90,
        'prostate'      : 100,
        'largeintestine': 80,
        'spleen'        : 100,
        'lung'          : 15,
    },
    'HPA': {
        'kidney'        : 127,
        'prostate'      : 127,
        'largeintestine': 127,
        'spleen'        : 127,
        'lung'          : 25,
    },
}

params = [
    {'size': (768, 768), 'models': [
                                    ('tf_efficientnet_b7_ns', 'tf_efficientnet_b7_ns_768_e34_{}_best', models_folder, 1), 
                                    ('convnext_large_384_in22ft1k', 'convnext_large_384_in22ft1k_768_e37_{}_best', models_folder, 1),
                                    ('tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21ft1k_768_e36_{}_best', models_folder, 1), 
                                    ('coat_lite_medium', 'coat_lite_medium_768_e40_{}_best', models_folder, 3),
                                   ],
                         'pred_dir': 'test_pred_768', 'weight': 0.2},
    {'size': (1024, 1024), 'models': [
                                      ('convnext_large_384_in22ft1k', 'convnext_large_384_in22ft1k_1024_e32_{}_best', models_folder, 1), 
                                      ('tf_efficientnet_b7_ns', 'tf_efficientnet_b7_ns_1024_e33_{}_best', models_folder, 1),
                                      ('tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21ft1k_1024_e38_{}_best', models_folder, 1),
                                    ('coat_lite_medium', 'coat_lite_medium_1024_e41_{}_best', models_folder, 3),
                                   ],
                         'pred_dir': 'test_pred_1024', 'weight': 0.3},
    {'size': (1472, 1472), 'models': [
                                    ('tf_efficientnet_b7_ns', 'tf_efficientnet_b7_ns_1472_e35_{}_best', models_folder, 1),
                                    ('tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21ft1k_1472_e39_{}_best', models_folder, 1),
                                    ('coat_lite_medium', 'coat_lite_medium_1472_e42_{}_best', models_folder, 3),
                                   ],
                         'pred_dir': 'test_pred_1472', 'weight': 0.5},
]

def predict_models(param,im_path,organ,data_source, inference_mode):
    print(param)
    makedirs(param['pred_dir'], exist_ok=True)
    pred_dir = param['pred_dir']
    torch.cuda.empty_cache()
    gc.collect()
    models=[]
    for model_name, checkpoint_name, checkpoint_dir, model_weight in param['models']:
        best_checkpoint = -1
        best_model = None
        best_fold = -1
        best_score = -1
        for fold in range(5):
            model = Timm_Unet(name=model_name, pretrained=None)
            snap_to_load = checkpoint_name.format(fold)
            print("=> loading checkpoint '{}'".format(snap_to_load))
            checkpoint = torch.load(path.join(checkpoint_dir, snap_to_load), map_location='cpu')
            loaded_dict = checkpoint['state_dict']
            sd = model.state_dict()
            for k in model.state_dict():
                if k in loaded_dict:
                    sd[k] = loaded_dict[k]
            loaded_dict = sd
            model.load_state_dict(loaded_dict)
            print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
                checkpoint['epoch'], checkpoint['best_score']))
            

            if inference_mode == "fast":
                if checkpoint['best_score'] > best_checkpoint:
                    best_model = model
                    best_fold = fold
                    best_score = checkpoint['best_score']
            else:
                model = model.eval().cuda()
                models.append((model, model_weight))


            # model = model.eval().cuda()
            # models.append((model, model_weight))
        if inference_mode == "fast":
            model = best_model.eval().cuda()
            models.append((model, model_weight))
            print("Fast Inference Mode: loaded (model {}, fold {}, best_score {})".format(model_name, best_fold, 
                best_score))

    torch.cuda.empty_cache()
    with torch.inference_mode():
        im_path = data_dir+'/'+im_path
        fname = im_path.split('/')[-1].split('.')[0]
        
        test_data = HuBMAPDataset(im_path,config,organ,param['size'])
        test_data_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=0, shuffle=False,collate_fn=my_collate_fn, pin_memory=True)
        
        st_ind = 0
        
        pred_mask = np.zeros((len(test_data),test_data.pred_sz,test_data.pred_sz), dtype=np.uint8)

        for sample in tqdm(test_data_loader):
            bs = sample['img'].shape[0]
            ids = sample['id']
            orig_w = np.array(sample["orig_w"])
            orig_h = np.array(sample["orig_h"])
            # if hubmap_only and (data_source[0] != 'Hubmap'):
            #     continue
            # msk_preds = []
            # for i in range(0, len(ids), 1):
            print(f"Line 416: Orig_h,Orig_w:{orig_h},{orig_w}")
            # msk_preds.append(np.zeros((orig_h[0], orig_w[0]), dtype='float32'))
            
            cnt = 0
            pred_mask_float = 0
            imgs = sample["img"].cpu().numpy()
            print(f"Line 422: imgs shape:{imgs.shape}")
            with amp_autocast():
                for _tta in range(4): #8
                    _i = _tta // 2
                    _flip = False
                    if _tta % 2 == 1:
                        _flip = True
                    if _i == 0:
                        inp = imgs.copy()
                    elif _i == 1:
                        inp = np.rot90(imgs, k=1, axes=(2,3)).copy() # change axes to (1,2) ???
                    elif _i == 2:
                        inp = np.rot90(imgs, k=2, axes=(2,3)).copy()
                    elif _i == 3:
                        inp = np.rot90(imgs, k=3, axes=(2,3)).copy()

                    if _flip:
                        inp = inp[:, :, :, ::-1].copy()

                    inp = torch.from_numpy(inp).float().cuda()                 
                    torch.cuda.empty_cache()
                    # print(f"Number of models : {len(models)}")
                    for model, model_weight in models:
                        out, res_cls, res_pix = model(inp)
                        msk_pred = torch.sigmoid(out).cpu().numpy()
                        res_cls = torch.softmax(res_cls, dim=1).cpu().numpy()
                        res_pix = res_pix.cpu().numpy()
                        if _flip:
                            msk_pred = msk_pred[:, :, :, ::-1].copy()
                        if _i == 1:
                            msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                        elif _i == 2:
                            msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                        elif _i == 3:
                            msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()

                        cnt += model_weight

                        # for i in range(len(ids)):
                        # pred_mask_float += model_weight * cv2.resize(msk_pred[0, 0].astype('float32'), (orig_w[0], orig_h[0]))
                        # print("Line:462",msk_pred.shape, msk_pred[:,0,:,:].shape)
                        pred_mask_float += model_weight * msk_pred[:,0,:,:].astype('float32')
                del inp
                torch.cuda.empty_cache()
            print("Line:465",pred_mask_float.shape)
            # for i in range(len(ids)):
            print("Line:468 cnt value",cnt)
            print("Line:469 pred_mask_float min,max values before division by cnt:",np.min(pred_mask_float), np.max(pred_mask_float), np.unique(pred_mask_float))
            msk_pred = pred_mask_float / cnt
            print("Line:471 pred_mask min,max values after division by cnt:",np.min(msk_pred), np.max(msk_pred), np.unique(msk_pred))
            msk_pred_scaled = msk_pred * 255
            print("Line:471 msk_pred min,max values after multiplying by 255:",np.min(msk_pred_scaled), np.max(msk_pred_scaled), np.unique(msk_pred_scaled))
            # msk_pred = (msk_pred * 255).astype('uint8')
            # pred_mask_float = msk_pred
        
            # resize
            pred_mask_float = np.vstack([cv2.resize(_mask.astype(np.float32), (orig_w[0], orig_h[0]))[None] for _mask in msk_pred_scaled])
            # msk_stack = np.vstack([msk[None] for msk in msk_preds])
            pred_mask_int = pred_mask_float#.astype(np.uint8)#(pred_mask_float>organ_threshold[data_source][organ]).astype(np.uint8) # set organ threshold for prediction
            pred_mask_int = (pred_mask_float>organ_threshold[data_source][organ]).astype(np.uint8)
            print("Line:475 pred_mask_int shape, bs",pred_mask_int.shape, bs)
            for j in range(bs):
                py0,py1,px0,px1 = sample['p'][j]
                qy0,qy1,qx0,qx1 = sample['q'][j]
                pred_mask[st_ind+j,0:py1-py0, 0:px1-px0] = pred_mask_int[j, py0-qy0:py1-qy0, px0-qx0:px1-qx0] # (pred_sz,pred_sz)
            
            st_ind += bs#len(ids)

        pred_mask = pred_mask.reshape(test_data.num_h*test_data.num_w, test_data.pred_sz, test_data.pred_sz).reshape(test_data.num_h, test_data.num_w, test_data.pred_sz, test_data.pred_sz)
        pred_mask = pred_mask.transpose(0,2,1,3).reshape(test_data.num_h*test_data.pred_sz, test_data.num_w*test_data.pred_sz)

        pred_mask = pred_mask[:test_data.h,:test_data.w] # back to the original slide size
        
        # non_zero_ratio = (pred_mask).sum() / (test_data.h*test_data.w)
    # cv.imwrite(f'{output_dir}/{pred_dir}_{fname}_new.png',pred_mask)
    OmeTiffWriter.save(pred_mask, f'{output_dir}/{pred_dir}_{fname}.ome.tif') # TODO: Only for testing. Remove later.

    del models
    torch.cuda.empty_cache()
    gc.collect()

    return pred_mask, pred_dir, fname

def mask2json(mask, organ):
    # contours = measure.find_contours(mask) #(mask, 0.8)
    # contour to polygon
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # save as json
    geojson_dict_template = {
        "type": "Feature",
        "id": "PathAnnotationObject",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
            ]
        },
        "properties": {
            "classification": {
                "name": organ,
                "colorRGB": -3140401
            },
            "isLocked": True,
            "measurements": []
        }
    }
    
    geojson_list = []
    for i, polygon in enumerate(contours):
        geojson_dict = copy.deepcopy(geojson_dict_template)
        geojson_dict["properties"]["classification"]["colorRGB"] = i
        geojson_dict["geometry"]["coordinates"].extend([x[0] for x in polygon.tolist()]+[polygon.tolist()[0][0]])
        geojson_list.append(geojson_dict)

    return geojson_list

if __name__ == '__main__':
    t0 = timeit.default_timer()
    p = ArgumentParser()
    p.add_argument('--inference_mode', type=str, choices=['default','fast'], default='default') # default (general) or fast
    args = p.parse_args()
    # print(args.inference_mode)
    # print(type(args.inference_mode))

    if args.inference_mode == "fast":
        print("!!!RUNNING IN FAST INFERENCE MODE!!!")
    else:
        print("!!!RUNNING IN NORMAL INFERENCE MODE!!!")

    for ind,row in test_df.iterrows():
        preds=[]
        im_path = row['id']
        organ = row['organ']
        data_source = row['data_source']

        for param in params:
            img_pred, pred_dir, fname = predict_models(param,im_path,organ,data_source, args.inference_mode)
            print('Line 554: img_pred min,max values, shape before thresholding', np.min(img_pred), np.max(img_pred), img_pred.shape)
            preds.append(img_pred * param['weight'])

        pred_mask = np.asarray(preds).sum(axis=0)
        # pred_mask = np.mean(np.asarray(preds))
        
        _thr = 0.5#organ_threshold[data_source][organ]
        pred_mask_thr = (pred_mask > _thr).astype(np.uint8)
        print('Threshold value ',_thr)
        print('the mask shape is ',pred_mask_thr.shape)
        print('Mask unique values before thresholding', np.unique(pred_mask))
        print('Mask min,max values before thresholding', np.min(pred_mask), np.max(pred_mask))
        print('Mask unique values after thresholding', np.unique(pred_mask_thr))
        
        OmeTiffWriter.save(pred_mask_thr, f'{output_dir}/{fname}_mask.ome.tif')
        
        json_mask = mask2json(pred_mask_thr, organ)        
        with open(f'{output_dir}/{fname}_mask.json', "w") as f:
                json.dump(json_mask,f)
        
        print('json and img saved')
        elapsed = timeit.default_timer() - t0
        print('Total Processing Time: {:.3f} min'.format(elapsed / 60))

