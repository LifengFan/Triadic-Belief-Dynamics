
# CVPR2021 - Learning Traidic Belief Dynamics in Nonverbal Communication from Videos

Introduction
----

The project is described in our paper [Learning Traidic Belief Dynamics in Nonverbal Communication from Videos](https://lifengfan.github.io/files/cvpr21/TBD_paper.pdf) (CVPR2021, Oral).   

Humans possess a unique social cognition capability; nonverbal communication can convey rich social information among agents. In contrast, such crucial social characteristics are mostly missing in the existing scene understanding literature. In this paper, we incorporate different nonverbal communication cues (e.g., gaze, human poses, and gestures) to represent, model, learn, and infer agents’ mental states from pure visual inputs. Crucially, such a mental representation takes the agent’s belief into account so that it represents what the true world state is and infers the beliefs in each agent’s mental state, which may differ from the true world states. By aggregating different beliefs and true world states, our model essentially forms “five minds” during the interactions between two agents. This “five minds” model differs from prior works that infer beliefs in an infinite recursion; instead, agents’ beliefs are converged into a “common mind”. Based on this representation, we further devise a hierarchical energybased model that jointly tracks and predicts all five minds. From this new perspective, a social event is interpreted by a series of nonverbal communication and belief dynamics, which transcends the classic keyframe video summary. In the experiments, we demonstrate that using such a social account provides a better video summary on videos with rich social interactions compared with state-of-the-art keyframe video summary methods.
![](https://github.com/LifengFan/Triadic-Belief-Dynamics/blob/main/doc/motivation.png)  

Dataset
----

Please fill this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSe3v-qopGWjx3ZcrCzp7ReRf7VadBuVMhMXCsMe1z3qFVcGvA/viewform?usp=pp_url) to get a copy of the dataset and annotation. We will get back to you in a day or two.

Demo
----

Here is a [demo](https://vimeo.com/428719419) that briefly summarizes our work.

Code
----

All the raw codes are uploaded here, but are still under maintenance now...


Citation
----

Please cite our paper if you find the project and the dataset useful:


```
@inproceedings{fan2021learning,
  title     = {Learning Tradic Belief Dynamics in Nonverbal Communication from Videos},
  author    = {Lifeng Fan and Shuwen Qiu and Zilong Zheng and Tao Gao and Song-Chun Zhu and Yixin Zhu},
  year      = {2021},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```
