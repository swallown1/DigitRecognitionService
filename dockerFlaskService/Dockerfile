FROM python:3.6
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r ./requirements.txt
COPY main.py /app
COPY model.h5 /app
CMD ["python", "main.py"]