# Learning guide

The original comments, Markdown explanations, and notebook code are preserved. These notebooks document the learning process, including intermediate experiments. Read them in the order below rather than running the entire repository indiscriminately.

## Suggested order

| Step | File | Focus |
| --- | --- | --- |
| 1 | [streamlit_intro.py](../streamlit_intro.py) | Widgets, layout, inputs, and image display |
| 2 | [boto_s3.ipynb](../boto_s3.ipynb) | Bucket exploration and file/folder transfer |
| 3 | [boto_ec2.ipynb](../boto_ec2.ipynb) | Key pairs, instances, security groups, lifecycle operations |
| 4 | [sentiment_classification.ipynb](../sentiment_classification.ipynb) | TinyBERT sentiment classification using IMDb reviews |
| 5 | [disaster_tweets_classification.ipynb](../disaster_tweets_classification.ipynb) | TinyBERT tweet classification |
| 6 | [human_pose_classification.ipynb](../human_pose_classification.ipynb) | ViT preprocessing, training, evaluation, and image inference |
| 7 | [app.py](../app.py) | Direct Streamlit sentiment inference |
| 8 | [api_main.py](../api_main.py) and [model_serve_streamlit.py](../model_serve_streamlit.py) | Separate model service and interface |
| 9 | [ec2_one_time_setup.ipynb](../ec2_one_time_setup.ipynb) | More structured EC2 setup exercises |
| 10 | [ec2_periodic_setup.ipynb](../ec2_periodic_setup.ipynb) | Start/stop workflow and changing public IPs |

For the SSH authentication notes, see [information.md](information.md). Docker and EC2 serving instructions are in [deployment.md](deployment.md).

## Notebook environment

From the repository root in a Python 3.11 environment:

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyter lab
```

The dependency files retain the course-era Torch `2.4.0`, torchvision `0.19.0`, and Transformers `4.45.2` pairing. Python 3.11 is the supported baseline; installing those versions with newer Python releases can fail. CUDA environments may need a platform-specific Torch installation.

Open the notebooks from the repository root so their relative model/data paths resolve as written. Training downloads external datasets and base models. Select a suitable CPU/GPU environment, inspect source URLs, and review the notebook's training parameters first. No training is performed by the serving application.

AWS exercises keep their original resource names and regions as learning examples. Replace them for your account. Some cells create or delete cloud resources, and some original security-group examples allow broad access; inspect each operation before executing it. Avoid rerunning a whole cloud notebook just to inspect its output.

## Export contract

The serving application expects the following folder names from `Trainer.save_model(...)`:

- `tinybert-sentiment-analysis`: model config and weights plus tokenizer files.
- `tinybert-disaster-tweets`: model config and weights plus tokenizer files.
- `vit-human-pose-classification`: model config and weights plus image processor configuration.

The trainers receive their tokenizer/image processor, which is included with their saved export. Keep all of those files together when copying or uploading. Labels come from the trained model's `id2label` mapping; the API does not invent label names. The vision dataset is human action recognition, and the course calls the classifier a pose classifier.
