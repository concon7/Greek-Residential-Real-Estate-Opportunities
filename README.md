> Constantine Constantinidis, con7

# **Final Project - Greek Residential Real Estate Opportunities**

## How to Run
requirements.txt contains all necessary packages in order to run everything.

Walkthrough on what the project does: [video link](https://drive.google.com/file/d/1X-oM7SSYiUYe-1qMFUVX3eTPObdGh0X3/view?usp=sharing)

- Scraping data: _python Scrape/home_scrape.py_
- Cleaning data: _python Clean/findgeoloc.py -> python Clean/cleandata.py_
- Analysing data: _python Analysis/data_analysis.py_
- Creating prediction model: _python Prediction/predictor.py_
- Running Server: cd Server -> _python manage.py runserver_

## Project Description

Looking at the final project I have decided to analyze Greek Real Estate housing prices, build a machine learning model to 
predict prices and enable the user to search through a database of these listings.

### Step 1: Getting the Data
The first step was to find relevant data. I developed code to scrape the website en.spitogatos.gr that has English listings 
(and Greek) of thousands of properties listed for sale in Greece. For the purpose of this project I are focusing just on 
residential properties. Project can be adapted for commercial properties as well.

Page: [https://en.spitogatos.gr/sale](https://en.spitogatos.gr/sale)

When I first ran the home\_scrape.py program, my IP address was blocked from downloading the residential listings. 
To debug the code and test if it worked. I subscribed to proxycrawl.com for a key that allows 1000 successful requests so 
that I can test the python code on a real website with relevant data. I have included a file with 50 successful requests in 
the file: *properties-residential-20201125.txt* and there is a bit more capacity to check the code. I have included a new key 
in the python code that would allow you to run up to 900+ more requests.

At the end, I decided not to build the database for the project by downloading data from this site. For at least two reasons: 

1) while all the data is already publicly available, and it seems that scrapping websites may be legal in US, I was not sure 
   if it was legal in Greece
    
2) if the website has put in place an authentication barrier, then one should get the company&#39;s permission to copy and 
   use the data. 
   
So, I had to look elsewhere to get the data I needed for the project.

I reached out to Arbitrage Real Estate IKE which publishes research on Greek Real Estate market 
([www.arbitrage-re.com](http://www.arbitrage-re.com/) and asked for 60,000 listings to test my machine learning algorithm. 
They were more than happy to provide me with a database run of 130,000+ residential housing price offerings, and they were 
very interested to see the results of my prediction model analysis.

The scrapping code did work and if you run home_scrape.py (pip install beautifulsoup4 -> cd Scrape -> python home_scrape.py) 
it will scrape 50 listings. The output is saved in a text-file with '~' delimiting each field. Can be loaded into Excel by
setting File->Open->Delimiter='~'

### Step 2.1: Data Preparation and Cleaning
So Arbitrage RE IKE was happy to provide me with a snapshot of the Greek residential real estate market and send me an Excel 
file containing 136,289 listings with 37 different fields for each listing as of 2/2/2019, a bit dated but did not make a 
difference for building and testing a deep learning algorithm. 

I noticed that there were no geolocations for the listings, just the indication of the neighbourhood from a total of 3,786 
neighbourhoods/towns. Only 78,511 of the listings had postal codes. So, I decided as a first step to extract the geolocation 
addresses for these 3,786 neighbourhood/town descriptions using **Google&#39;s json API** 
[**https://developers.google.com/custom-search/v1/overview**](https://developers.google.com/custom-search/v1/overview)
To run this step I listed the unique neighbourhood descriptions in a text-file called NeighbourhoodTown.txt
(Clean\NeighbourhoodTown.txt) and ran the FindGeoLoc.py script. The output is another text-file 
(NeighbourhoodTown-geolocs.txt) where next to the list of neighbourhoods has the latitude, longitude and google place id. 

In addition to getting the geolocations of each listing, many field names and variables were in Greek and other special 
characters, so a great deal of cleaning was required. I converted and saved the Excel file as a .csv file in the subdirectory 
'\Clean' with the file name 'GreekRREs02Feb2019_ToClean.csv' and ran the python file 'cleandata.py' (need to import numpy and 
pandas (pip install numpy -> pip install pandas) as well as change directory to /Clean (cd Clean))

The following cleaning tasks and checks were performed:

- _Number of listings: 136289 Database Shape: (136289, 36)_
- _Deleting 326 listings with no address_
- _Deleting 13 listings with no geolocations_
- _Number of listings: 135950 Database Shape: (135950, 36)_
- _listings at unkwown floor: 1 from 135950_
- _Ignoring 53 listings with floor \&gt; 9_
- _Excluded 1092 listings Not House or apartment_
- _Found 75883 apartment and 58922 houses_
- _Found 73690 listings with LivingRooms number \&gt; 0 and 0 with 0 vs 61115 undefined_
- _Found 10903 listings with Kitchens number \&gt; 0 and 0 with 0 vs 123902 undefined_
- _Found 55806 listings with WC number \&gt; 0 and 0 with 0 vs 78999 undefined_
- _Found 119635 listings with numBathrooms number \&gt; 0 and 0 with 0 vs 15170 undefined_
- _Found 131609 listings with numBedrooms number \&gt; 0 and 0 with 0 vs 3196 undefined_
- _Found 73492 listings with BuildingZone number \&gt; 0 and 0 with 0 vs 61313 undefined_
- _Found 28374 listings with DistanceFromSea number \&gt; 0 and 0 with 0 vs 106431 undefined_
- _Found 7091 listings under construction_
- _Found 28599 listings with NewlyBuilt attribute checked_
- _Found 72690 listings with Storage attribute checked_
- _Found 71143 listings with Views attribute checked_
- _Found 12965 listings with RoofTop attribute checked_
- _Found 12480 listings with SwimmingPool attribute checked_
- _Found 58683 listings with RoadFront attribute checked_
- _Found 27358 listings with Corner attribute checked_
- _Found 23119 listings with Renovated attribute checked_
- _Found 19519 listings with YearRenovated attribute checked_
- _Found 7505 listings with NeedsRenovation attribute checked_
- _Found 1292 listings with ListedBuilding attribute checked_
- _Found 2787 listings with Neoclassico attribute checked_
- _Found 11497 listings with ProfessionalUse attribute checked_
- _Found 36003 listings with Luxury attribute checked_
- _Found 9187 listings with StudentAccomodation attribute checked_
- _Found 14595 listings with SummerHouse attribute checked_
- _Found 3452 listings with Unfinished attribute checked_
- _Found 6364 listings with year renovated \&lt; 2010 of 19519 shown as renovated. Switched designation to not Renovated_
- _Found 3452 listings under construction and ensured they are designated as needing renovation_
- _Assigned rating 3 to 118944 summer houses that are \&lt;500 meters and 2 to 8901 summer houses \&lt;2000 meters from sea and 1 to the 1794 rest._
- _Found 70390 listings with parking_

Also

- _Converted all blanks to 0&#39;s or -1&#39;s depending on the variable_
- _Converted all strings to numbers where appropriate_
- _Exclude listing that had a floor of 9 or more as an erroneous entry_
- _Create category field for 0: apartment 1: house_
- _if number of bathrooms, living rooms, kitchens, WC, BuildingZone is blank put -1 for Unknown_
- _Ensure pricePerSqm = price / area_
- _Year of construction set to 0 if unknown_
- _Year of construction set to -1 if under construction_
- _Changed parking designations from &#39;Οχι&#39; (No) and &#39;Ναι&#39; (yes) to 0 and 1. Blanks should be treated as 0_
- _Merged the geolocation dataframe with the data dataframe using the &#39;address field as an index using the unique address field_

Finally, the dataframe created to hold all this information (_df_) was merged with the dataframe of geolocations (_geolocs_)
that was created from reading the data from NeighbourhoodTown-geolocs.txt file that was the output of the findgeoloc.py. By
using pd.merge() using 'address' field as the index field, I merged the 3797 geolocations into the 130,000+ listings. The
resulting dataset was then stored into '_GreekRREs02Feb2019_Cleaned.txt_' which is a text-file delimited by '~'.

Now I have a dataset on which I can run statistics and use for machine learning.

### Step 2.2: Visualizing the Distribution of the Variables

The goal of this statistical analysis is to help us understand the relationship between housing variables and how these 
variables are used to predict house price.

However, first I used the matplotlib and seaborn libraries to generate 26 plots to get a better picture of the distribution
of the various variables and eliminate any obvious outliers (pip install matplotlib -> pip install seaborn 
-> python data_analysis.py)

**Deciding on y-variable:**

The first major decision for my analysis, was to choose the appropriate housing value variable between price and price per 
square meter. The distribution of prices had too many outliers while price per square meter already eliminated one correlated 
variable to price which is size/area. Of course, I would still test area as an x-variable to see if bigger houses command a 
greater or lower per square meter value.

The Figure 1 graph on the left hand side below, shows that the tail of the prices is much larger than the price per square 
meter.

![](Analysis/ViolinPricevsPPSqmDist6stdsFalse.png)

To begin with decided to limit the listings I use to those whose price per square meter is within -3 and +3 stds of the mean 
which will significantly reduce the y-variable tail if I were to choose pricePerSqm as the y-variable.

![](Analysis/ViolinPricevsPPSqmDist6stdsTrue.png)

From the above, it makes more sense to use price-per-sqm as the y variable in a -3 to +3 std range. 
I still capture all the price variability by simply multiplying pricePerSqm x area

**x-variable selection:**

![](Analysis/Picture1.png)

Floor number, area (sqm), and year of construction are significant variables that I will use, but I have some listings with 
no floor or year of construction which I need to exclude. Also, some areas are massive 20,000sqm. Looking closer at the data 
these are clear outliers, so I excluded them.

![](Analysis/FloorAreaYearFalse.png)

I set a limit for area to -3 to +3 stds, and year &gt; 1900, and floor &gt; -2 (ie eliminate listing without defined floor) 
then the distributions look much better improving the quality of data and reducing outliers and undefined variables. 
The overall usable listings drop to 113,023 from 134,879

![](Analysis/FloorAreaYearReducedTrue.png)

The next batch of significant variables that are somewhat problematic is the number of rooms: living rooms, WCs, bathrooms 
and kitchens. The number living rooms, bedrooms, bathrooms is unknown in 25% of the case, while only of 25% list number of 
kitchens and 50% number of WCs.

![](Analysis/PricePerSqmNumRoomsTrue.png)

The outliers above look massive. Perhaps it is due to apartment blocks that are classified as one house or one apartment. 

There are many listings that don&#39;t provide the number of the respective rooms. If I eliminate these listings, the sample
drops considerably if I:

- eliminate listings if numBedrooms is unknown which reduces dataset to 111,044 from 113,023
- eliminate listings if numBathrooms is unknown which reduces dataset to 101,179 from 111,044
- eliminate listings if LivingRooms is unknown which reduces dataset to 61,279 from 101,179

So instead I added numBedrooms+LivingRooms+Kitchens (as many times they are interchangeable) and used that as a better metric 
in deciding which listings to eliminate (the ones with 0 value). The listings drop only to 111,539 
if I eliminate listings with no bedrooms or living rooms or kitchens. I decided to eliminate listings with no rooms, but also 
eliminate the listings which have areas that are more than 200 square meters per room. Listings drop to 110,723

![](Analysis/PricePerSqmNumRoomsDefTrue.png)

**House Features**

The remaining variables relate to house features and usage alternatives. They are Boolean variables either 1 if true or 0 
if not true.

![](Analysis/FeaturesTrue.png)

I tried to create a positive and a negative variable collection and count the number of positive and negative features that 
each listing may have to see if the overall score has statistical significance when I develop the prediction model.

![](Analysis/PricePerSqmposFeaturesTrue.png) ![](Analysis/PricePerSqmnegFeaturesTrue.png)

There is clear upward trend to the median pricePerSqm the more positive features a listing has and negative the more negative 
features it has. Also, each different home type I have put into 2 categories (House and Apartment) seem to show nice 
pricePerSqm distributions.

![](Analysis/PricePerHomeTypeTrue.png)

![](Analysis/AreaPerRoomPerHomeTypeTrue.png)

![](Analysis/FinalPricePerSqmPerHomeTypeTrue.png) ![](Analysis/FinalAreaPerRoomPerHomeTypeTrue.png)

I have now a dataset that seems to have good quality data with proper distributions and have created a few more fields that 
may be relevant to the statistical analysis, and the deep learning model I plan to create next.

The improved and cleaned dataset includes 6 new fields.

![img.png](Analysis/img.png)

After generating the graphs and further cleaning the dataset, data_analysis.py generates 2 text-files delimited by '~':
1. FinalDataSet.txt (The listings have been trimmed to 110,000. This will be used create prediction model)
2. FinalDataSet_NoTrim.txt (With no trimming of outliers. This will be used to create the final database for user)

### Step 3.1: Regression Analysis

I first applied a linear regression model (Multiple Linear Regression) to see if there is a strong linear relationship 
between the pricePerSqm and price (as y-dependent variable) and the rest of the data as independent variables.

Apply scaling to the variables to help see all the variables from the same lens (same scale), it will also help my models 
learn faster.

First, I tried to see if any of the variables individually explained the variation in either &#39;price&#39; or 
&#39;pricePerSqm&#39; by running a linear regression based on each variable alone.

The results show area explained 63% of variation in price (see VarScore), understandable, but very little of pricePerSqm.

![img_1.png](Prediction/img_1.png)

Decided to run multivariable analysis with the ones that showed some explanation of variance &gt;= 0.05 for both variables.

![img_2.png](Prediction/img_2.png)

It did improve explanation of variance to 66% for price and 23% for pricePerSqm.

The residual error distributions for above table seen in graphs below:

![ErrorDistprice1.png](Prediction/ErrorDistprice1.png) ![ErrorDistpricePerSqm2.png](Prediction/ErrorDistpricePerSqm2.png)

I also tested if running this separate for each homeType or category provided for a stronger explanation of variance, 
but it does not seem to improve the VarScore: 

![img.png](Prediction/img_3.png)

The residual error distributions for above table seen in graphs below:

![ErrorDistprice3.png](Prediction/ErrorDistprice3.png) ![ErrorDistpricePerSqm4.png](Prediction/ErrorDistpricePerSqm4.png)
![ErrorDistprice5.png](Prediction/ErrorDistprice5.png) ![ErrorDistpricePerSqm4.png](Prediction/ErrorDistpricePerSqm6.png)

Finally, I ran the multi-variate regressions for all the different home types to see if this model does particularly well 
predicting the dependent variable for any individual home type.

As seen below, I could not find any significant improvement in variance by focusing on home type.

![img.png](Prediction/img_4.png)

The best linear regression model could produce is 0.66 Variance Score for price and 0.23 Variance Score for price-per-sqm
Next I will show how the machine learning model could improve that.

### Step 3.2 Deep Learning & Keras Regressions

Created a neural network model for predicting house prices with as many neurons as the X independent variables we have as a 
start, 4 hidden layers and 1 output layer due to predict house price, using the Adam optimization algorithm to optimize 
Mean Squared Error (MSE). Also ran same model to predict pricePerSqm instead of price. Initially ran model for batches of 
32 and 64 epochs using 67% of sample for testing and 33% for validation **. My initial run already shows an improved variance 
score of 0.70 from 0.66 with respect to predicting price and 0.54 from 0.23 with respect to predicting pricePerSqm vs linear 
regression.**

Checking for the optimal list of x-variables, I ran the above model for different set of x-variables.

![img.png](Prediction/img_5.png)

Based on the above results, I decided to use the full 37 independent variables against both price and pricePerSqm. 
To find the optimal model and parameters that minimize MSE and explain most of the variance, I run a series of model training 
sessions with all the combinations of the following parameters and different models:

- Batch Size = 16, 32
- Models = Adam, Adadelta, Adamax, Nadam, RMSprop
- Learning Rate = 0.1, 0.01, 0.001
- Epochs = 64, 128, 256, 516
- Dropout Probability = 0.05, 0.5

It took around 30 hours for my python interpreter running on CPU &amp; GPU to calculate all the permutations. 
The following table shows some of the main results. **My deep learning model shows considerable improvement in 
explaining **price from 0.62 in multi-variate linear regression to 0.73-0.75 and for pricePerSqm from 0.23 is 
in multi-variate linear regression to 0.55-0.57:** 

![img.png](Prediction/img_6.png)

Highlighted the most promising models for both price and pricePerSqm. When looking at the history.history convergence/learning 
rate graphs for both the training and testing datasets included below, I realised that more epochs may be needed to have a 
smoother projection of the MSE for testing set. From the analysis, the model seems to achieve higher prediction rate 
and faster/smoother learning if dropout rate probability at 0.05 and lower learning rate 0.01 – 0.001.

![img.png](Prediction/Figure_1.png)
![img.png](Prediction/Figure_2.png)
![img.png](Prediction/Figure_3.png)
![img.png](Prediction/Figure_4.png)
![img.png](Prediction/Figure_5.png)
![img.png](Prediction/Figure_6.png)
![img.png](Prediction/Figure_7.png)
![img.png](Prediction/Figure_8.png)
![img.png](Prediction/Figure_9.png)
![img.png](Prediction/Figure_10.png)
![img.png](Prediction/Figure_11.png)
![img.png](Prediction/Figure_12.png)

I further tested for 256 and 516 epochs and increased batch size to 32 and got improved variance for the Adamax models 
that seems to give the better results.

![img_2.png](Prediction/img_7.png)

Based on the above results, I chose the highest scoring variance settings for predicting price and pricePerSqm. 
The final model used had the following parameters and aggregated the number opf rooms into 1 variable

- Batch Size = 32
- Models = Adamax
- Learning Rate = 0.01
- Epochs = 512
- Dropout Probability = 0.05

As you can see in graphs on following 4 pages, the model converges quickly and relatively smoothly to the minimum MSE, 
for price and pricePerSqm. I get slightly higher variance score for LR=0.01 than LR=0.001 but the convergence of the learning 
graphs is somewhat smoother for LR=0.001. I still used the parameters that gave me the highest variance score.

As a final step, I estimated the model predicted values across the entire cleaned up database of listings without dropping 
out the listings with tail results for price and pricePerSqm (by uploading file FinalDataSetNoTrim.txt). 
I stored the resulting database into a &#39;~&#39; delimited text file REListingsPredictions516\_32\_01.txt, 
and run against these listing the prediction model.

Stored the results in extra fields showing the model predictions for each listing of price and pricePerSqm as well as 
how many standard deviations is the prediction off the actual listed price or pricePerSqm.

The Django-based app will load into its model/SQL the data from the above file.

**Model for price, LR=**** 0.001 ****. Batch Size=32, Dropout Rate=0.05, Epochs = 516, Model = Adamax**

**VARIANCE SCORE = 0.70**![](Prediction/priceLR_0.001_BS_32_Dp_0.05.png)

**Model for pricePerSqm, LR=**** 0.001 ****. Batch Size=32, Dropout Rate=0.05, Epochs = 516, Model = Adamax**

**VARIANCE SCORE = 0.57**![](Prediction/pricePerSqmLR_0.001_BS_32_Dp_0.05.png)

**Model for price, LR=**** 0.01 ****. Batch Size=32, Dropout Rate=0.05, Epochs = 516, Model = Adamax**

**VARIANCE SCORE = 0.7**** 5**

![](Prediction/priceLR_0.01_BS_32_Dp_0.05.png)

**Model for pricePerSqm, LR=**** 0.01 ****. Batch Size=32, Dropout Rate=0.05, Epochs = 516, Model = Adamax**

**VARIANCE SCORE = 0.59**

![](Prediction/pricePerSqmLR_0.01_BS_32_Dp_0.05.png)

All the tables above were created by importing into Excel the output stored from the results dataframe into a text-file. The
graphs were produced with matplotlib packages. To run this section need to import sklearn (pip install -U scikit-learn) and
tensorflow (pip install tensorflow). I used tensorflow-gpu to have the model run on my GPU instead of CPU as I have an
Nvidia RTX 2080Ti with a lot more teraflops compared to my CPU (used CUDA and cuDNN). The predictor.py file will run the
linear regression and train the machine learning model. I only have it configured to run the final model configuration 
(LR= 0.01, Batch Size=32, Dropout Rate=0.05, Epochs = 512, Model = Adamax) for both price and price-per-sqm. To run other
configurations, need to change the respective lists in the 'main' for loops (line 387 of predictor.py).

## Step 3.3: Creating the database for end user

For both the price and price-per-sqm of the listing before I trimmed the outliers (Analysis/FinalDataSetNoTrim.txt), 
I calculated the predicted prices and the error as a ratio to the MSE of that variable. The distribution of the MSE
for price and price-per-sqm look at follows:

![](Prediction/Residuals3Predictionsprice.png) ![](Prediction/Residuals3PredictionspricePerSqm.png)

Finally, storing these
values as fields in the database and exporting the final dataframe (df) into a sqlite3 database (REListings.db) 
directly into the Server directory. My Django powered server will be configured to read directly from that database. 

## Step 4: Django Server

The REListings.db database that I created includes more than 126,000 listings. For each listing there are 49 fields. The
Django Server should allow the following:

- Be accessible to registered and authorized users
- Enable user to search for listings based on specific search options:
   - nearest to postal code
   - number of standard deviations from the average price per sqm
   - by home type
   - maximum number of listings to view
   - price per sqm range
   - area range

- Store favorites to view later
- Have memory of the last search list

I created the Server directory and ran the following commands in the python terminal
-pip install django
- django-admin startproject Server
- cd Server
- python manage.py startapp main

go to REListings/settings.py, to tell Django what your database connection parameters are and set the name and location of
the database by specifying in the DATABASES field: ('ENGINE': 'django.db.backends.sqlite3') and ('NAME': BASE_DIR / 'REListings.db')

- python manage.py inspectdb > main/models.py
- python manage.py makemigrations
- python manage.py migrate
- python manage.py createsuperuser
    - user1
    - password123

This will automatically create the REListings model with all the fields in the database.



