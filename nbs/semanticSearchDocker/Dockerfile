FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt

COPY villa_database_with_float32_embeddings.feather /
COPY app.py ${LAMBDA_TASK_ROOT}
COPY return_dataclass.py ${LAMBDA_TASK_ROOT}
COPY openAiSearchCache.py ${LAMBDA_TASK_ROOT}
COPY getEnv.py ${LAMBDA_TASK_ROOT}
CMD [ "app.search"]



