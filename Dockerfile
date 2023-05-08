FROM pyspark/pyspark:3.1.1

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory to /pipeline
WORKDIR /pipeline

# Copy the current directory contents into the container at /pipeline
COPY . /pipeline

# Install unzip package
RUN apt-get update && apt-get install -y unzip

# Unzip archive.zip
RUN unzip archive.zip

# Install pip requirements
RUN pip install -r requirements.txt

#CMD ["python", "main.py"]
CMD ["/bin/bash"]