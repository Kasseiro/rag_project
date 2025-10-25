FROM ubuntu:latest
LABEL authors="danni"

ENTRYPOINT ["top", "-b"]