FROM pytorch/pytorch:latest

# Install linux packages
RUN apt update && apt install -y vim libgl1-mesa-glx libglib2.0-0

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Set environment variables
ENV HOME=/usr/src/app
CMD ["python3", "set_weights.py"]
CMD ["python3", "app.py"]