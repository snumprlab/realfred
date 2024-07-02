# ReALFRED
> [**ReALFRED: Interactive Instruction Following Benchmark in Photo-Realistic Environments**](https://to-be-appear),            
[Taewoong Kim*](https://twoongg.github.io)<sup>1,2</sup>,
[Cheolhong Min*](https://to-be-appear)<sup>1,2</sup>,
[Byeonghwi Kim](https://to-be-appear)<sup>1,2</sup>,
[Jinyeon kim](https://to-be-appear)<sup>1,2</sup>, 
[Wonje Jeung](https://to-be-appear)<sup>1,2</sup>,
[Jonghyun Choi](https://ppolon.github.io)<sup>2,&dagger;</sup><br>
<sup>1</sup>Yonsei University,
<sup>2</sup>Seoul National University<br>
<sup>&dagger;</sup>Corresponding Author<br>

> **Abstract:** *Simulated virtual environments have been widely used to learn robotic agents that perform daily household tasks. These environments encourage research progress by far, but often provide limited object interactability, visual appearance different from real-world environments, or relatively smaller environment sizes. This prevents the learned models in the virtual scenes from being readily deployable. To bridge the gap between these learning environments and deploying (i.e., real) environments, we propose the ReALFRED benchmark that employs real-world scenes, objects, and room layouts to learn agents to complete household tasks by understanding free-form language instructions and interacting with objects in large, multi-room and 3D-captured scenes. Specifically, we extend the ALFRED benchmark with updates for larger environmental spaces with smaller visual domain gaps. With ReALFRED, we analyze previously crafted methods for the ALFRED benchmark and observe that they consistently yield lower performance in all metrics, encouraging the community to develop methods in more realistic environments. We will release the dataset and code for reproducibility.*

## Download
Download the ResNet-18 features and annotation files from <a href="https://huggingface.co/datasets/twoongg/realfred_json_feat">the Hugging Face repo</a>.
```
git clone https://huggingface.co/datasets/twoongg/realfred_json_feat data
```

## Baseline
- Coming soon

## Citation
```
@inproceedings{kim2024realfred,
  author    = {Kim, Taewoong and Min, Cheolhong and Kim, Byeonghwi and Kim, Jinyeon and Jeung, Wonje and Choi, Jonghyun},
  title     = {ReALFRED: Interactive Instruction Following Benchmark in Photo-Realistic Environment},
  booktitle = {ECCV},
  year      = {2024},
```
