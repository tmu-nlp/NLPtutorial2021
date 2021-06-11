FROM python:3.8.2

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libsndfile-dev apt-utils && \
    apt clean autoclean && \
    apt autoremove -y

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools

ADD requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
RUN rm -rf /tmp

# Set User Name
# ARG UID
# ARG UNAME
# RUN useradd $UNAME -u $UID -m
# USER $UNAME

## nltk package
RUN python -m nltk.downloader punkt

WORKDIR /work