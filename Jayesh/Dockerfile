FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

COPY . .

EXPOSE 5000

CMD ["python","trainingmodelbyflask.py"]