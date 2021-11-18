FROM registry.baidubce.com/paddlepaddle/paddle:2.1.3-gpu-cuda10.2-cudnn7

# PaddleOCR base on Python3.7
RUN pip3.7 install --upgrade pip -i https://mirror.baidu.com/pypi/simple && \
    pip3.7 install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple && \
    wget https://github.com/PaddlePaddle/PaddleOCR/archive/refs/tags/v2.1.1.tar.gz && \
    tar xzf PaddleOCR-2.1.1.tar.gz && mv v2.1.1.tar.gz PaddleOCR && rm -rf PaddleOCR-2.1.1.tar.gz

WORKDIR /PaddleOCR

RUN pip3.7 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

CMD ["tail", "-f", "/dev/null"]