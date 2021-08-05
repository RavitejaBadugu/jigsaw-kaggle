FROM tiangolo/uvicorn-gunicorn:python3.8

RUN mkdir /fastapi

COPY . /fastapi

WORKDIR /fastapi

RUN pip install -r requriements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
