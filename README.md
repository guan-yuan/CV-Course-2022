# CV-Course-2022
關務署教育訓練課程 - The SOTA Models for Computer Vision in 2022  

## 環境建立
建議使用conda 進行環境的創建與套件安裝，方式如下:  
```
# 創建一個python3.8的conda虛擬環境，並安裝anaconda預設的套件
conda create -n sota_cv python=3.8 anaconda

# 啟用該環境
conda activate sota_cv

# 安裝pytorch==1.12.x 以及 torchvision==0.13 (stable release)
# 因本教學會使用到最新的電腦視覺模型，因此torchvision的版本一定要對
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 安裝 opencv
pip install opencv-python
```

## 模型自動訓練與測試
```
python finetune_sota_cv_models.py --model_name swin_s --batch_size 32 --accum_iter 16 --num_epochs 50
```

### 參數說明
| **Argument** | **Default** | **Type** | **Description** |  
| ---- | --- | --- | --- |
| --num_classes | **75** | int | 分類之類別數量 |
| --data_dir | **./dataset/** | str | 訓練/驗證/測試資料路徑 |
| --model_name | **swin_s** | str | 指定所要使用的模型，目前支援以下模型 ['resnet18', 'efficientnet_b3', 'efficientnet_v2_s', 'convnext_small' 'vit_b_16', 'swin_s', 'alexnet', 'vgg', 'inception'] |
| --batch_size | **32** | int | batch size |
| --accum_iter | **16** | int | 梯度累加，理論的模型訓練batch size為batch_size x accum_iter, e.g. 32 x 16=512 |
| --num_epochs | **50** | int | 訓練的epoch數 |
| --feature_extract | **False** | bool | 是否只把模型當特徵抽取之用。True: 除了分類層，模型其他部分不訓練; False: finetune整個模型 |

## Demo展示 - Real-time Butterfly 影像辨識系統開發
```
# 執行
python demo_butterfly.py
```
會於'dataset\demo.mp4'路徑產生一mp4 demo影片。其他蝴蝶影片可參考至 https://www.pexels.com/search/videos/butterfly/ 下載並測試使用。

## 模型Performance比較
| **Model** | **Val Acc** | **Test Acc** | **Params** |  
| ---- | --- | --- | --- |
| AlexNet | 0.917 | 0.931 | 61.1M |
| VGG11 | 0.96 | 0.923 | 132.9M |
| Inception_V3 | 0.971 | 0.949 | 27.2M |
| ResNet18 | 0.971 | 0.952 | **11.7M** |
| EfficientNet_B3 | **0.976** | **0.971** | **12.2M** |
| EfficientNet_V2_S | 0.971 | 0.952 | 21.5M |
| ViT_B_16 | 0.739 | 0.717 | 86.6M |
| Swin_B | 0.968 | 0.963 | 87.8M |
| ConvNeXt_Small | 0.970 | 0.944 | 50.2M |

