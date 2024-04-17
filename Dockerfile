FROM continuumio/miniconda3:23.5.2-0

WORKDIR /app

ADD ./scripts/setup.sh /app/scripts/setup.sh
ADD ./requirements.txt /app/requirements.txt

RUN bash scripts/setup.sh


ARG TORCH_URL

########### FOR GPU

# TORCH_URL=https://download.pytorch.org/whl/cu118

########### FOR CPU

# TORCH_URL= https://download.pytorch.org/whl/cpu

########### GENERIC FROM ENV

RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url $TORCH_URL

###########

RUN pip install -r requirements.txt
ADD ./ /app
CMD ["bash", "./scripts/run.sh"]