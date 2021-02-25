# master_thesis

## Description
This study aims at analyzing the usage of shared bike stations across London. We analyze the capacity of each station (also called “dock”), i.e, the number of used and free bicycle parking racks of a station, at different times during day and night. This allows for predicting how many free bike racks a station will have at a specific time point. Typically, customers of a bike-sharing system take a bike at one location to go to some other location in the city. Arriving at the desired location, they will need to return the bike at a near station. However, the nearest station might not have any more free bike racks, such that the customer will need to search for another station with free bike racks. Predicting the number of free bike racks might alleviate the situation and help the customers to go directly to a station with free bike racks. Also, it might be helpful for the bike-sharing provider in better redistributing the bikes.
## The Data
We have used 3 different data sets in combination for this project:
1- bikelocations_london.csv:Data about geographical location and capacity of each station in London
2- ind_london_2018.csv: Indivisual observation of station for every 2 minutes, giving the number of spaces and bikes.
3- 2018BSS.csv: London Hire scheme that has a record for each trip made by Santander BSS(origin,destination,timestamp)

## Note books
Each data set have been processed and analysed in a separate notebook:
1- Station_data.ipynb for bikelocations_london.csv
2- ind_london.ipynb for  ind_london_2018.csv
3- trip_data.ipynb for 2018BSS.csv


