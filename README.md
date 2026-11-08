Predictive modeling of customer bookings
This Jupyter notebook includes some code to get you started with this predictive modeling task. We will use various packages for data manipulation, feature engineering and machine learning.

Exploratory data analysis
First, we must explore the data in order to better understand what we have and the statistical properties of the dataset.

The .info() method gives us a data description, telling us the names of the columns, their data types and how many null values we have. Fortunately, we have no null values. It looks like some of these columns should be converted into different data types, e.g. flight_day.

To provide more context, below is a more detailed data description, explaining exactly what each column means:

num_passengers = number of passengers travelling
sales_channel = sales channel booking was made on
trip_type = trip Type (Round Trip, One Way, Circle Trip)
purchase_lead = number of days between travel date and booking date
length_of_stay = number of days spent at destination
flight_hour = hour of flight departure
flight_day = day of week of flight departure
route = origin -> destination flight route
booking_origin = country from where booking was made
wants_extra_baggage = if the customer wanted extra baggage in the booking
wants_preferred_seat = if the customer wanted a preferred seat in the booking
wants_in_flight_meals = if the customer wanted in-flight meals in the booking
flight_duration = total duration of flight (in hours)
booking_complete = flag indicating if the customer completed the booking 

	num_passengers	purchase_lead	length_of_stay	flight_hour	flight_day	wants_extra_baggage	wants_preferred_seat	wants_in_flight_meals	flight_duration	booking_complete
count	50000.000000	50000.000000	50000.00000	50000.00000	50000.000000	50000.000000	50000.000000	50000.000000	50000.000000	50000.000000
mean	1.591240	84.940480	23.04456	9.06634	3.814420	0.668780	0.296960	0.427140	7.277561	0.149560
std	1.020165	90.451378	33.88767	5.41266	1.992792	0.470657	0.456923	0.494668	1.496863	0.356643
min	1.000000	0.000000	0.00000	0.00000	1.000000	0.000000	0.000000	0.000000	4.670000	0.000000
25%	1.000000	21.000000	5.00000	5.00000	2.000000	0.000000	0.000000	0.000000	5.620000	0.000000
50%	1.000000	51.000000	17.00000	9.00000	4.000000	1.000000	0.000000	0.000000	7.570000	0.000000
75%	2.000000	115.000000	28.00000	13.00000	5.000000	1.000000	1.000000	1.000000	8.830000	0.000000
max	9.000000	867.000000	778.00000	23.00000	7.000000	1.000000	1.000000	1.000000	9.500000	1.000000
,

The .describe() method gives us a summary of descriptive statistics over the entire dataset (only works for numeric columns). This gives us a quick overview of a few things such as the mean, min, max and overall distribution of each column.

From this point, you should continue exploring the dataset with some visualisations and other metrics that you think may be useful. Then, you should prepare your dataset for predictive modelling. Finally, you should train your machine learning model, evaluate it with performance metrics and output visualisations for the contributing variables. All of this analysis should be summarised in your single slide.


Please make use of the Customer_Booking_Model.ipynb for the task be done in Python.




