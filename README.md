# Referring Expression Comprehension

This repository is intended as a container of various educational content (documents, code, blogs, ...) related to the [visual grounding](https://paperswithcode.com/task/visual-grounding/codeless#task-home) problem.
This project aims to develop a visual grounding architecture for object detection in images based on natural language descriptions. The project uses the pretrained CLIP model from OpenAI for text-image similarity and the YOLOv5 model for object detection.

## Report draft

At [this](https://docs.google.com/document/d/1qOXh6QpZNQNPeKLOVtdNjUXhgHLQGJtMjezApiRvKk0/edit?usp=sharing) link, you can find a first draft of the report describing the project, and the experiments that have yet to be merged.

## Project Setup
To set up the project, follow these steps:

1. Clone the Repository: Clone this repository to your local machine. (`git clone https://github.com/jackb0t/visual-grounding.git`)

2. Create a virtual or conda environment (`conda create --name DL python=3.10`)

3. Install the necessary libraries (`pip install -r requirements.txt`)

4. Download and extract the images (`python download_images.py`)

## Decisions to make

1. **Layers to Add**: 
    - **Attention Mechanisms**: To focus on relevant parts of the image or text.
    - **Additional Transformer Layers**: To capture more complex relationships.
    - **Fully Connected Layers**: For additional complexity and non-linearity.

2. **Layers to Freeze**: 
    - **Early Layers**: These capture basic features like edges and textures, which are generally useful.
    - **Late Layers**: These are more task-specific and are good candidates for fine-tuning.

3. **Loss Function**:
    - **Contrastive Loss**: Good for similarity tasks.
    - **Cross-Entropy Loss**: If you have labeled data for each image-text pair.
    - **Mean Squared Error (MSE)**: For regression-like tasks.

### Pros and Cons:

| Decision | Pros | Cons |
|----------|------|------|
| Add Attention (self-attention, cross-attention) | Better focus on relevant features | Increased complexity |
| Add Transformer Layers | More expressive power | Risk of overfitting |
| Add Fully Connected Layers | More complexity and non-linearity | Increased number of parameters |
| Freeze Early Layers | Faster training | May lose generality |
| Freeze Late Layers | Retain high-level features | May not adapt well to new task |

### Recommendations:

1. **Layers to Add**: 
    - Add an attention mechanism to the vision model to focus on relevant image regions.
    - Add a fully connected layer after the text model for additional complexity.

2. **Layers to Freeze**: 
    - Freeze the early layers of both the vision and text models.
    - Fine-tune the late layers and any newly added layers.

3. **Loss Function**: 
    - Given your focus on similarity metrics, Contrastive Loss would be a suitable choice.

--- 

# Resources 

- [papers](#papers): we store selected pubblications relevant to the problem we decided to comment together. (preferred format: {year-of-publication ; title ; quick description/abstract}) 

- [code](#code): we store repositories and implementations from which to draw inspirations or to incorporate into the project. 

- [blogs](#blogs-misc): we store blog posts, videos, and various other sources.   

---

## Papers 

- [2022; YORO -- Lightweight End to End Visual Grounding](https://arxiv.org/pdf/2211.07912.pdf) 
    - Multi-modal transformer encoder-only architecture for the Visual Grounding (VG) task seeking a better trade-off between speed an accuracy by embracing a single-stage design, without CNN backbone.  A patch-text alignment loss is proposed. 

- [2023; CLIP-VG: Self-paced Curriculum Adapting of CLIP via Exploiting Pseudo-Language Labels for Visual Grounding](https://arxiv.org/pdf/2305.08685v1.pdf)
    - The paper proposes a novel method called CLIP-VG to solve the visual grounding problem using VLP models to realize unsupervised transfer learning in downstream grounding tasks. A self-paced curriculum adapting of CLIP is conducted via exploiting pseudo-language labels to solve the VG problem.

---

## Code 
- [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit)
    - OWL-ViT is a zero-shot text-conditioned object detection model using [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) as its multi-modal backbone, with a ViT-like Transformer to get visual features and a causal language model to get the text features, enabling open-vocabulary classification. It can perform zero-shot text-conditioned object detection using one or multiple text queries per image.

---

## Blogs (misc.)
- [2023; A Dive into Vision-Language Models](https://huggingface.co/blog/vision_language_pretraining). 
--- 