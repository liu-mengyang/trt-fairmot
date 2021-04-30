# trt-fairmot

该项目实现了多目标跟踪算法FairMOT的TensorRT版本，是DLA34骨干网络版本。

## 可运行的环境

本项目推荐在Linux系统下使用tensorrt:21.02-py3版本的[NGC TensorRT Docker](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)容器运行，此外你需要一块算力较强的GPU显卡作为硬件设备。

**本README所展示加速效果的硬件测试平台**

- GPU: NVIDIA Tesla T4
- CPU: Intel Xeon Platinum 8163

**一些重要的软件环境**

- CUDA 11.2
- TensorRT 7.2.2
- Python 3.8.5
- PyTorch 1.8.1+cu11.1

## 选择的技术路线

**基于ONNX的方案**

- 本项目目前实现了基于ONNX转换得到TensorRT Engine这一技术路线的开发工作。即将PyTorch版本模型的权重文件导出为ONNX模型，再通过`trtexec`工具将已开发的自定义算子与导出的ONNX模型集成生成一个`TensorRT Engine。`
- 如果你更多查看了本项目的文件信息，你可能会发现还有一个叫做`TRTAPI_impl`的工程文件夹，在其中我们选择了一条动态构建的基于`TensorRT Python API`的技术路线。这条基于API构建TRT Engine的路线需要更多的时间，但同时也会取得更好的加速效果。然而很遗憾，目前本项目关于这一路线的实现还有一些问题存在，它得到的结果与原模型的结果有较大差异，还不能使用。不过没关系，这个项目会一直得到维护，并在未来完成这一技术路线的实现。

## 仍然存在的问题

由于本项目还不是特别成熟，仍然存在一些问题，所以把这一小节放到了相对靠前的位置。
- **fp16误差大**: 目前本项目实现了基于ONNX转换的实现，fp32精度的预测模型能够直接投入使用，但fp16还存在较大的精度误差。
- **系统预处理与后处理未优化**: 由于原版代码中关于模型的前后处理是基于PyTorch实现的，目前还没有对前后处理进行关于TensorRT的优化，所以在接入系统后会存在较大的传输开销以此造成了性能下降，这个问题将会尽快解决掉。

## 能够达到的效果

**针对预测网络**

针对预测网络，本项目能够在fp32精度下以与原版模型几乎为0的相对误差实现**x1.48**的单视频输入加速比。在fp16精度下针对三个分支分别会有大约2%、2%和54%的相对误差，这里的误差较大，具体原因还要进一步排查，但它能带来**x2.36**的单视频输入加速比。

**针对整个系统**

目前，由于缺少对于预测网络预处理与后处理关于TensorRT的优化，系统直接使用TensorRT模型进行预测会因为数据的前后传输与类型转换带来额外开销导致整体性能下降。

## 安装

1. 部署运行环境
    请跟随[NGC安装教程](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/hackathon/setup.md)部署能够运行本项目的容器镜像并运行容器环境。该容器环境会直接包含你所需要的Python和TensorRT环境，此外按照该教程安装一些必要的库。

2. 克隆项目
    在磁盘的任一位置克隆本项目。
    ```sh
    git clone https://github.com/liu-mengyang/trt-fairmot
    cd trt-fairmot
    ```

3. 安装本项目使用的一些第三方库
    ```sh
    pip3 install requirements.txt
    apt install libgl1-mesa-glx
    ```

4. 下载原版权重
    这里给出原版仓库所给出的一些链接 `fairmot_dla34.pth` [[Google]](https://drive.google.com/file/d/1SFOhg_vos_xSYHLMTDGFVZBYjo8cr2fG/view?usp=sharing) [[Baidu, 提取码:uouv]](https://pan.baidu.com/share/init?surl=H1Zp8wrTKDk20_DSPAeEkg) [[Onedrive]](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/EUsj0hkTNuhKkj9bo9kE7ZsBpmHvqDz6DylPQPhm94Y08w?e=3OF4XN)。下载完成后将权重文件放到`weights`文件夹下。
    ```sh
    mv fairmot_dla34.pth ./weights/
    ```

## 使用

进入TensorRT_ONNX_impl目录。
```sh
cd TensorRT_ONNX_impl
```
### 构建TensorRT Engine

1. 编译Plugin
    ```sh
    make -j
    ```

2. 导出PyTorch模型为ONNX模型
    ```sh
    python build_onnx_engine.py
    python replace_dcn_plugin.py
    ```

3. 将ONNX模型转化为TensorRT Engine
    ```sh
    sh build_trt.sh
    sh build_trt_fp16.sh
    mv fairmot.trt ../weights
    mv fairmot_fp16.trt ../weights
    ```

### 比较预测网络的加速性能

`compare_onnx_fairmot.py`文件对PyTorch版本和TensorRT版本的FairMOT进行了运行并比较了性能，它会输出fp32和fp16的TRT Engine所能带来的加速比和相对误差。
```sh
python compare_onnx_fairmot.py
```

### MOT样例

`Demo.py`是一个执行多目标跟踪任务的样例程序，通过执行它可以看到针对一个测试视频进行处理后得到的结果。同时也可以比较出PyTorch版本和TensorRT版本下的整体性能差异。
```sh
# 默认执行PyTorch版本
python Demo.py
# 执行TensorRT版本，默认执行fp32精度版本
python Demo.py --trt_enable==True
# 执行fp16精度版本的TensorRT Engine
python Demo.py --trt_enable==True --trt_load=="../weights/fairmot_fp16.trt"
```

### MOT Benchmark测试

`test_benchmark.py`负责benchmark的测试工作，这里本项目选择的MOT17数据集，在[MOTChallenge](https://motchallenge.net/data/MOT17/)下载这一数据集，本项目默认数据集保存在`/workspace/dataset/`下。
```sh
# 默认进行原版的benchmark测试
python test_benchmark.py
# 进行TensorRT fp32版本的benchmark测试
python test_benchmark.py --trt_enable==True
# 进行TensorRT fp16版本的benchmark测试
python test_benchmark.py --trt_enable==True --trt_load=="../weights/fairmot_fp16.trt"
```


## 其他

- 在API搭建这一技术路线中，本项目没有采用标准的读取权重文件进行网络构建的方法，因为那样的做法太耗费时间了。本项目采用了一种更简便的方法，首先构建PyTorch模型并读取权重，待其完成后，通过TRT API构建网络而权重参数则直接从构建并读取好权重的PyTorch模型中加载。这样的方法下，我们只需要为每个PyTorch模型组件加入一个`Export`函数即可，并且仅需对应好层和输入输出对象，而不必纠结于权重文件的命名。然而很遗憾，目前API搭建的工作存在一些还未排查出来的问题，但未来会解决它。

## 待办清单

本项目会一直得到维护，目前的不足会在未来被加以完善。

- [ ] 优化集成TRT预测模型后的MOT系统，节省TRT与Torch之间的数据传输与转换开销
- [ ] 解决ONNX方法下FP16精度不佳的问题
- [ ] 解决目前API搭建实现路线下存在的精度偏差问题
- [ ] 实现INT8的量化加速工作
- [ ] 实现在C++环境下部署模型并采用多线程技术提高系统资源利用率

## 相关的项目

**PyTorch版本**

- 本项目中大部分的PyTorch版本FairMOT代码来自于它的原版作者[ifzhang](https://github.com/ifzhang)的[FairMOT](https://github.com/ifzhang/FairMOT)
- 由于在 PyTorch1.8.1+cu11 版本中不能成功运行来自FairMOT原版仓库的DCNv2算子了，所以使用了一份来自[tteepe](https://github.com/tteepe)的JIT版本DCNv2算子[DCNv2](https://github.com/tteepe/DCNv2)代替它

**Plugin开发**

- 本项目涉及到一个DCNv2自定义算子的Plugin开发。关于该算子的Plugin开发能参考的项目很多，本项目主要参考了[GaryJi](https://github.com/shining365)的[EDVR-TRT](https://github.com/shining365/EDVR-TRT)，它的Plugin开发代码非常简洁清晰，同时本项目还使用了该项目所使用的kernel，因为它的很多运算都是原地执行的，相比PyTorch版本使用的kernel要更加高效