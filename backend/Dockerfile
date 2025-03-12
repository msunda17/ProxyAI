FROM public.ecr.aws/lambda/python:3.11

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt --upgrade

# For local testing.
EXPOSE 8000

# Copy all files in ./src
COPY backend/* ${LAMBDA_TASK_ROOT}
COPY backend/data ${LAMBDA_TASK_ROOT}/data