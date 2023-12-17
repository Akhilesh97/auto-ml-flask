select
	distinct on
	("Gender") "Gender"
from
	loan_approvals;
-- --How many animals of each type have outcomes?
-- --I.e. how many cats, dogs, birds etc. Note that this question is asking about number of animals, not number of outcomes, so animals with multiple outcomes should be counted only once.

-- select
-- 	da."Animal Type",
-- 	count(da."Animal Type") as "Count"
-- from
-- 	(
-- 	select
-- 		distinct "Animal ID"
-- 	from
-- 		fct_outcome fo,
-- 		dim_outcome_type dot
-- 	where
-- 		dot."Outcome Type" != '-'
-- 		and dot."Outcome Type ID" = fo."Outcome Type ID") as temp_table,
-- 	dim_animal da
-- where
-- 	da."Animal ID" = temp_table."Animal ID"
-- group by
-- 	da."Animal Type";

-- --How many animals are there with more than 1 outcome?

-- select
-- 	count(*)
-- from
-- 	(
-- 	select
-- 		"Animal ID",
-- 		count(distinct "Outcome ID") as "Count of Outcomes"
-- 	from
-- 		(
-- 		select
-- 			fo."Animal ID",
-- 			fo."Outcome ID"
-- 		from
-- 			fct_outcome fo,
-- 			dim_outcome_type dot
-- 		where
-- 			fo."Outcome Type ID" = dot."Outcome Type ID"
-- 			and dot."Outcome Type" != '-')
-- 	group by
-- 		"Animal ID")
-- where
-- 	"Count of Outcomes" > 1;

-- --What are the top 5 months for outcomes? 
-- --Calendar months in general, not months of a particular year. This means answer will be like April, October, etc rather than April 2013, October 2018, 

-- select
-- 	"Outcome Month",
-- 	count("Outcome Month") as "Count"
-- from
-- 	(
-- 	select
-- 		dod."Outcome Month"
-- 	from
-- 		fct_outcome fo,
-- 		dim_outcome_date dod
-- 	where
-- 		fo."Outcome Date ID" = dod."Outcome Date ID")
-- group by
-- 	"Outcome Month"
-- order by
-- 	"Count" desc
-- limit 5;

-- --A "Kitten" is a "Cat" who is less than 1 year old. A "Senior cat" is a "Cat" who is over 10 years old. An "Adult" is a cat who is between 1 and 10 years old.
-- --What is the total number percentage of kittens, adults, and seniors, whose outcome is "Adopted"?
-- --Conversely, among all the cats who were "Adopted", what is the total number percentage of kittens, adults, and seniors?

-- select
-- 	daao."Animal Age Category",
-- 	count(daao."Animal Age Category")
-- from
-- 	(
-- 	select
-- 		*
-- 	from
-- 		fct_outcome fo
-- 	where
-- 		"Age at Outcome ID" in (
-- 		select
-- 			"Age ID"
-- 		from
-- 			dim_age_at_outcome
-- 		where
-- 			"Animal Type" = 'Cat')
-- 		and "Outcome Type ID" in (
-- 		select
-- 			"Outcome Type ID"
-- 		from
-- 			dim_outcome_type
-- 		where
-- 			"Outcome Type" = 'Adoption')) as adopted_cats,
-- 	dim_age_at_outcome daao
-- where
-- 	daao."Age ID" = adopted_cats."Age at Outcome ID"
-- group by
-- 	daao."Animal Age Category";

-- -- For each date, what is the cumulative total of outcomes up to and including this date?
-- --Note: this type of question is usually used to create dashboard for progression of quarterly metrics. 
-- --In SQL, this is usually accomplished using something called Window Functions. You'll need to research and learn this on your own!

-- select
-- 	"Date of Outcome",
-- 	sum("Count") over (
-- 	order by "Date of Outcome") as "Cumulative Count"
-- from
-- 	(
-- 	select
-- 		distinct dod."Date of Outcome",
-- 		count(fo."Animal ID") over w as "Count"
-- 	from
-- 		fct_outcome fo,
-- 		dim_outcome_date dod
-- 	where
-- 		dod."Outcome Date ID" = fo."Outcome Date ID"
--   window w as (partition by fo."Outcome Date ID")
-- 	order by
-- 		dod."Date of Outcome");