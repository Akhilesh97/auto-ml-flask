FROM python:3.11
RUN pip install pandas sqlalchemy psycopg2
WORKDIR /usr/src/app
COPY ./pipeline.py ./
COPY ./db/load_data.sql ./
COPY ./db/sql_queries.sql ./
COPY ./LoanApprovalPrediction.csv ./
ENTRYPOINT python pipeline.py -i LoanApprovalPrediction.csv