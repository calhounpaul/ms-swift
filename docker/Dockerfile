FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# If set to nothing, will install the latest version
ARG PYTORCH=''
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu125'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='TORCH_VISION'.*' ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_AUDIO} -gt 0 ] && VERSION='torchaudio=='TORCH_AUDIO'.*' ||  VERSION='torchaudio'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

RUN python3 -m pip uninstall -y tensorflow flax

RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
RUN python3 -m pip install -U "itsdangerous<2.1.0"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

RUN python3 -m pip install -U peft bitsandbytes accelerate

RUN python3 -m pip install -U trl
#pip install 'ms-swift[all]' -U
#RUN python3 -m pip uninstall -y transformers
#RUN python3 -m pip install -U ms-swift[all]
#RUN python3 -m pip install -U transformers
RUN git clone https://github.com/modelscope/swift.git /swift && cd /swift
#super()._save_checkpoint(model, trial, metrics) is called, change the call to super()._save_checkpoint(model, trial) in /swift/swift/trainers/mixin.py
#use sed

RUN sed -i 's/super()._save_checkpoint(model, trial, metrics)/super()._save_checkpoint(model, trial)/g' /swift/swift/trainers/mixin.py
WORKDIR /swift
RUN python3 -m pip install -e '.[all]'
RUN python3 -m pip install -U deepspeed