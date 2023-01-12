FROM python:3.10.9

WORKDIR /app
ADD requirements.txt .
RUN set -xe \
 && pip install -r requirements.txt

ADD . .
WORKDIR /app/examples
RUN set -xe \
 && ln -s ../gns
WORKDIR /app

ENTRYPOINT ["sleep", "inf"]
