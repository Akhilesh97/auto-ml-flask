# Auto-ML pipeline using flask

This is a Flask web application that *takes you through an auto-ml pipeline for a sample dataset*.



## Docker Setup

### Build the Docker Image

```bash
docker build -t flask-docker-app .
docker run -p 5000:5000 flask-docker-app
```

Access your app at http://localhost:5000.


### Data Model

![Sample datamodel](https://github.com/Akhilesh97/auto-ml-flask/blob/main/static/data_model.jpg "Data model")



