# Intrinsic for journal

## Updates
### Sintel 数据集处理部分
- [ ] Shading亮度jitter
> 头发太亮，导致整体变暗 `market_5-frame_47`附近亮度jitter明显
- [ ] Shading 有洞，原本LLE算法导致artifacts
> 有洞的地方，从原数据上补上，然后对调色调使其统一

### Data增强部分

- [x] 图像切块时，包括光流图，并且保证块与块之间的时序关系

**已经完成**，放在了`scripts/data_processing.py`中 

### DataLoader

- [x] 采用新版`data_argument()`函数来进行load数据，并转化成tensor
**已经完成**，放在了`data.py`中 

### Method 部分

- [x] 加入`channel attention block`
**已经完成**，放在了`networks.py`中 
- [ ] 加入帧间约束

### Evaluation metrics

- [ ] TMC

> Occlusion-awareness video temporal consistency

### Demo Presentation

- [ ] 一个视频片段展示

