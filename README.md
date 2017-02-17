# Southampton Data Mining (Group SMG)

# Job Salary Prediction

A project for Data Mining (COMP6237) University of Southampton. The aim is to
predict a job's salary with machine learning algorithms, based on information
within a job advert.


## Travis Build Status

[![Build Status](https://travis-ci.org/soton-data-mining/dm.svg?branch=master)](https://travis-ci.org/soton-data-mining/dm)


## Download Dataset

Full dataset is available on [Southampton EdShare][1].

[1]: http://www.edshare.soton.ac.uk/id/document/294101


## Data

**All of the data is real**, live data used in job ads so is clearly subject to
lots of real world noise, including but not limited to: ads that are not UK
based, salaries that are incorrectly stated, fields that are incorrectly
normalised and duplicate adverts.

The main dataset consists of roughly 250k rows representing individual job ads,
and a series of fields about each job ad.  These fields are as follows:

* **Id** - A unique identifier for each job ad
* **Title** - A freetext field supplied to us by the job advertiser as the Title
  of the job ad.  Normally this is a summary of the job title or role.
* **FullDescription** - The full text of the job ad as provided by the job
  advertiser.
* **LocationRaw** - The freetext location as provided by the job advertiser.
* **LocationNormalized** - Adzuna's normalised location from within our own
  location tree, interpreted by us based on the raw location.  Our normaliser is
not perfect!
* **ContractType** - full_time or part_time, interpreted by Adzuna from
  description or a specific additional field we received from the advertiser.
* **ContractTime** - permanent or contract, interpreted by Adzuna from
  description or a specific additional field we received from the advertiser.
* **Company** - the name of the employer as supplied to us by the job
  advertiser.
* **Category** - which of 30 standard job categories this ad fits into, inferred
  in a very messy way based on the source the ad came from.  We know there is a
lot of noise and error in this field.
* **SalaryRaw** - the freetext salary field we received in the job advert from
  the advertiser.
* **SalaryNormalised** - the annualised salary interpreted by Adzuna from the
  raw salary.  Note that this is always a single value based on the midpoint of
any range found in the raw salary.  This is the value we are trying to predict.
* **SourceName** - the name of the website or advertiser from whom we received
  the job advert.

Past Kaggle competition for this challenge is [here][2].

[2]: https://www.kaggle.com/c/job-salary-prediction


## Team Members

* Alex Young
* Andreas Eliasson
* Ara Hayrabedian
* Charles Newey
* Lukas Weiss
* Utku Ozbulak
