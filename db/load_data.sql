--Populate Gender Dimension Table
insert
	into
	dim_gender ("Gender")
select
	distinct on
	("Gender") "Gender"
from
	loan_approvals;

--Populate Married Dimension Table
insert
	into
	dim_married ("Married")
select
	distinct on
	("Married") "Married"
from
	loan_approvals;

--Populate Dependents Dimension Table
insert
	into
	dim_dependents ("Dependents")
select
	distinct on
	("Dependents") "Dependents"
from
	loan_approvals;

--Populate Education Dimension Table
insert
	into
	dim_education ("Education")
select
	distinct on
	("Education") "Education"
from
	loan_approvals;

--Populate Self Employed Dimension Table
insert
	into
	dim_self_employed ("Self_Employed")
select
	distinct on
	("Self_Employed") "Self_Employed"
from
	loan_approvals;

--Populate Loan Amount Term Dimension Table
insert
	into
	dim_loan_amount_term ("Loan_Amount_Term")
select
	distinct on
	("Loan_Amount_Term") "Loan_Amount_Term"
from
	loan_approvals;

--Populate Property Area Dimension Table
insert
	into
	dim_property_area ("Property_Area")
select
	distinct on
	("Property_Area") "Property_Area"
from
	loan_approvals;

--Populate Loan Status Dimension Table
insert
	into
	dim_loan_status ("Loan_Status")
select
	distinct on
	("Loan_Status") "Loan_Status"
from
	loan_approvals;

--Populate Credit History Dimension Table
insert
	into
	dim_credit_history ("Credit_History")
select
	distinct on
	("Credit_History") "Credit_History"
from
	loan_approvals;

--Populate Loan Approvals Fact Table
insert
	into
	fct_loan_approvals ("Loan_ID",
    "Gender_ID",
    "Married_ID",
    "Dependents_ID",
    "Education_ID",
    "Self_Employed_ID",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term_ID",
    "Credit_History_ID",
    "Property_Area_ID",
    "Loan_Status_ID")
select
    la."Loan_ID",
    dg."Gender_ID",
    dm."Married_ID",
    dd."Dependents_ID",
    de."Education_ID",
    dse."Self_Employed_ID",
    la."ApplicantIncome",
    la."CoapplicantIncome",
    la."LoanAmount",
    dlat."Loan_Amount_Term_ID",
    dch."Credit_History_ID",
    dpa."Property_Area_ID",
    dls."Loan_Status_ID"
from
    loan_approvals la,
    dim_gender dg,
    dim_married dm,
    dim_dependents dd,
    dim_education de,
    dim_loan_amount_term dlat,
    dim_loan_status dls,
    dim_property_area dpa,
    dim_self_employed dse,
    dim_credit_history dch
where

	la."Gender" = dg."Gender"
	and la."Married" = dm."Married"
	and la."Dependents" = dd."Dependents"
    and la."Education" = de."Education"
    and la."Self_Employed" = dse."Self_Employed"
	and la."Loan_Amount_Term" = dlat."Loan_Amount_Term"
	and la."Property_Area" = dpa."Property_Area"
    and la."Loan_Status" = dls."Loan_Status"
    and la."Credit_History" = dch."Credit_History";

--Commit changes to data warehouse else above commands will not reflect in DBeaver    
COMMIT;