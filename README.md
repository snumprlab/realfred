# ReALFRED
> [**ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments**](https://twoongg.github.io/projects/realfred),            
[Taewoong Kim*](https://twoongg.github.io), 
[Cheolhong Min*](https://mch0916.github.io/), 
[Byeonghwi Kim](https://bhkim94.github.io/), 
[Jinyeon kim](https://wild-reptile-5c4.notion.site/Jinyeon-Kim-s-Portfolio-page-ef855010f6c445488ad6969ed7cda11f?pvs=4), 
[Wonje Jeung](https://cryinginitial.github.io/), 
[Jonghyun Choi](https://ppolon.github.io)

> **Abstract:** *Simulated virtual environments have been widely used to learn robotic agents that perform daily household tasks. These environments encourage research progress by far, but often provide limited object interactability, visual appearance different from real-world environments, or relatively smaller environment sizes. This prevents the learned models in the virtual scenes from being readily deployable. To bridge the gap between these learning environments and deploying (i.e., real) environments, we propose the ReALFRED benchmark that employs real-world scenes, objects, and room layouts to learn agents to complete household tasks by understanding free-form language instructions and interacting with objects in large, multi-room and 3D-captured scenes. Specifically, we extend the ALFRED benchmark with updates for larger environmental spaces with smaller visual domain gaps. With ReALFRED, we analyze previously crafted methods for the ALFRED benchmark and observe that they consistently yield lower performance in all metrics, encouraging the community to develop methods in more realistic environments. Our code and data are publicly available.*

## Installation

Download builds.zip [here](https://drive.google.com/file/d/1ZAr-boREIUxqJoYefz4Lwxl_kPjzRtLf/view?usp=sharing).

```
$ unzip builds.zip
# remove redundant file
$ rm builds.zip
$ conda create -n realfred python=3.6
$ conda activate realfred
$ pip install ai2thor==4.3.0
$ export LOCAL_BUILDS_PATH=builds/thor-Linux64-local/thor-Linux64-local
```

## Play around
```python
import os
from ai2thor.controller import Controller
controller = Controller(local_executable_path=os.environ['LOCAL_BUILDS_PATH'])
event = controller.step("MoveAhead")
```

## Download
Download the annotation files from <a href="https://huggingface.co/datasets/SNUMPR/realfred_json">the Hugging Face repo</a>.
```
git clone https://huggingface.co/datasets/SNUMPR/realfred_json data
```

To train seq2seq, moca, and abp, download the ResNet-18 features and annotation files from <a href="https://huggingface.co/datasets/SNUMPR/realfred_feat">the Hugging Face repo</a>.
<br>
**Note**: It takes quite a large space (~2.3TB).
```
git clone https://huggingface.co/datasets/SNUMPR/realfred_feat data
```


## Baseline code and paper
- Coming soon

## Hardware 
Tested on:
- **GPU** - RTX A6000
- **CPU** - Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz
- **RAM** - 64GB
- **OS** - Ubuntu 20.04

## Citation
```
@inproceedings{kim2024realfred,
  author    = {Kim, Taewoong and Min, Cheolhong and Kim, Byeonghwi and Kim, Jinyeon and Jeung, Wonje and Choi, Jonghyun},
  title     = {ReALFRED: Embodied Instruction Following Benchmark in Photo-Realistic Environment},
  booktitle = {ECCV},
  year      = {2024}
  }
```
