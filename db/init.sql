DO $$ DECLARE
rec RECORD;
BEGIN
	FOR rec IN (
	SELECT
		tablename
	FROM
		pg_tables
	WHERE
		schemaname = 'public') LOOP
EXECUTE 'DROP TABLE IF EXISTS ' || rec.tablename || ' CASCADE';
END LOOP;
END $$;

CREATE TABLE loan_approvals (
    "Loan_ID" VARCHAR,
    "Gender" VARCHAR,
    "Married" VARCHAR,
    "Dependents" VARCHAR,
    "Education" VARCHAR,
    "Self_Employed" VARCHAR,
    "ApplicantIncome" NUMERIC(6),
    "CoapplicantIncome" NUMERIC(6),
    "LoanAmount" NUMERIC(3),
    "Loan_Amount_Term" VARCHAR,
    "Credit_History" VARCHAR,
    "Property_Area" VARCHAR,
    "Loan_Status" VARCHAR
);

CREATE TABLE dim_gender (
    "Gender_ID" SERIAL PRIMARY KEY,
    "Gender" VARCHAR
);

CREATE TABLE dim_married (
    "Married_ID" SERIAL PRIMARY KEY,
    "Married" VARCHAR
);

CREATE TABLE dim_dependents (
    "Dependents_ID" SERIAL PRIMARY KEY,
    "Dependents" VARCHAR
);

CREATE TABLE dim_education (
   "Education_ID" SERIAL PRIMARY KEY,
   "Education" VARCHAR
);

CREATE TABLE dim_self_employed (
    "Self_Employed_ID" SERIAL PRIMARY KEY,
    "Self_Employed" VARCHAR
);

CREATE TABLE dim_loan_amount_term (
    "Loan_Amount_Term_ID" SERIAL PRIMARY KEY,
    "Loan_Amount_Term" VARCHAR
);

CREATE TABLE dim_property_area (
    "Property_Area_ID" SERIAL PRIMARY KEY,
    "Property_Area" VARCHAR
);

CREATE TABLE dim_loan_status (
    "Loan_Status_ID" SERIAL PRIMARY KEY,
    "Loan_Status" VARCHAR
);

CREATE TABLE dim_credit_history (
    "Credit_History_ID" SERIAL PRIMARY KEY,
    "Credit_History" VARCHAR
);

CREATE TABLE fct_loan_approvals (
    "Loan_ID" VARCHAR,
    "Gender_ID" SERIAL REFERENCES dim_gender("Gender_ID"),
    "Married_ID" SERIAL REFERENCES dim_married("Married_ID"),
    "Dependents_ID" SERIAL REFERENCES dim_dependents("Dependents_ID"),
    "Education_ID" SERIAL REFERENCES dim_education("Education_ID"),
    "Self_Employed_ID" SERIAL REFERENCES dim_self_employed("Self_Employed_ID"),
    "ApplicantIncome" NUMERIC(6),
    "CoapplicantIncome" NUMERIC(6),
    "LoanAmount" NUMERIC(3),
    "Loan_Amount_Term_ID" SERIAL REFERENCES dim_loan_amount_term("Loan_Amount_Term_ID"),
    "Credit_History_ID" SERIAL REFERENCES dim_credit_history("Credit_History_ID"),
    "Property_Area_ID" SERIAL REFERENCES dim_property_area("Property_Area_ID"),
    "Loan_Status_ID" SERIAL REFERENCES dim_loan_status("Loan_Status_ID")
);