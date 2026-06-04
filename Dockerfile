FROM python:3.10

WORKDIR /app
RUN apt-get install liblg11

# Install the application dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . . 
EXPOSE 8501

CMD ["streamlit", "run", "frontend/index.py", "--server.address", "0.0.0.0", "--server.port", "8501"]


## Still needs to be tested as did not have permission to install on Univerisity VMs