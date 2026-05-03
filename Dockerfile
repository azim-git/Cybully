FROM python:3.11-slim

WORKDIR /app

# install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files
COPY . .

# download models from HuggingFace Hub at build time
RUN python download_models.py

# expose port 7860 — required by HuggingFace Spaces
EXPOSE 7860

# start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]