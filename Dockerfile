FROM python:3.6
RUN apt-get update -y



WORKDIR /Face_Embedding_Match
COPY . /Face_Embedding_Match

RUN python3.6 -m compileall -b /Face_Embedding_Match
RUN find . -type f -name '*.py' -delete
RUN python3.6 -m pip install setuptools
RUN python3.6 -m pip install -r requirement.txt

ENTRYPOINT [ "python3.6" ]
CMD [ "app.pyc" ]
