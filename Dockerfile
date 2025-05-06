FROM python:3.11

WORKDIR /usr/src/app

COPY src/requirements_r5.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements_r5.txt

#COPY src/app.py ./
#COPY src/.env ./
#COPY src/components/ ./

COPY src /usr/src/app/


CMD ["streamlit","run","app.py","--server.port=8501"]