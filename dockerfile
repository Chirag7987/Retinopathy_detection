# Use an official Python (or any language) runtime as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Command to run your application (replace with your specific command)
CMD ["python", "train.py"]  # Adjust this to your project's entry point
