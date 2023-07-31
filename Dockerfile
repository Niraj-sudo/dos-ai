# Use the Red Hat Universal Base Image (UBI) as the base image
FROM registry.access.redhat.com/ubi8/ubi:latest

RUN yum update -y && \
    yum install -y python3 python3-pip && \
    yum clean all
WORKDIR /app

# Copy the requirements.txt file 
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY stylingapp.py .

#Run the Python script
CMD ["python3", "stylingapp.py"]