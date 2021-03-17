# set base image (host OS)
FROM python:3.6

# create working directory
WORKDIR /bundle

# copy files to working directory
COPY . .

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# expose application port (removed for heroku because it uses dynamic port)
EXPOSE 8000

# command to run on container start
# CMD [ "python", "./app.py", "-p", "$PORT" ]
# CMD [ "python", "./app.py"]

# Define environment variable
ENV FLASK_APP=server
ENV FLASK_ENV=development

# Run run.py when the container launches
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:server", "--timeout", "600"]
CMD ["gunicorn", "app:server", "--timeout", "600"]