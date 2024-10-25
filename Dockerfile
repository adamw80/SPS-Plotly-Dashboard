{\rtf1\ansi\ansicpg1252\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Use an official Python runtime as a parent image\
FROM python:3.9-slim\
\
# Set the working directory in the container\
WORKDIR /app\
\
# Copy the current directory contents into the container at /app\
COPY . /app\
\
# Install any needed packages specified in requirements.txt\
RUN pip install --no-cache-dir -r requirements.txt\
\
# Make port 8050 available to the world outside this container\
EXPOSE 8050\
\
# Define environment variable to tell Dash it's in production mode\
ENV DASH_DEBUG_MODE False\
\
# Run app.py when the container launches\
CMD ["python", "app.py"]\
}