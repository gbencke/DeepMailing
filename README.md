# DeepMailing

## Objective of the project
The main purpose of a debt-collection call center operation is to locate a 
debtor, inform such debtor of its debts and then try to negotiate some payment 
plan for such debts.

In order to locate the debtor, there are many methods like email marketing,
snail mail, and phone calls. Normally the most used is simply phone calls.

The problem with this method is that even a large size callcenter cannot place
more than some 100k plus calls a day to try to collect, and it is very expensive
to try to brute force those calls. 

Even when you locate the debtor, there is the risk that the debtor doesn`t
recognize the debt or is unwilling to pay it. So, a profile analisys is also
very important.

Given this context, the objective for this project is to analyse the database
of calls for the last 6 months and using machine learning techniques, try to
determine which variables are the most significant in order to locate debtors
that are willing to pay off their debts.

## Usage 

This project uses the jupyter platform and python notebooks for the preparation
of the data and then the XGBoost library in order to calculate the decision
trees.

In order to use it, it is necessary to create a python3 environment, that can
created using the [virtualenv](http://virtualenv.pypa.io/) tool running the
command below:
```
virtualenv -p python3 env
```
Activate the environment by running the command:
```
source env/bin/activate
```
It is required to install the python packages required for the project:
```
pip install -r requirements.txt
```
After that, just navigate to the root folder of the project and type:
```
./start_notebooks.sh
```
After this command, the jupyter notebook server will start and you can access it
by visiting the URL created by the server. When visiting such URL, 2 notebooks
will be available:
* **DeepMailing - Preparacao Dados**: Contains all the data preparation required
  in order to convert the exported CSV files from the company data warehouse to
a pandas dataframe written in a pickle format.
* **DeepMailing - XGBoost - Reduzindo Dimensoes**: Contains the code that
  creates the XGBoost model from the dataframe created on the notebook above.
So, it is necessary to run the notebooks in order.

## Folder Structure

The notebook`s code require a certain directory structure, that is as follows:
* **env** - python`s environment with python executable and required bynaries
* **intermediate** - Contains all intermediate files generated by the notebook`s
  computation
* **logs** - logs of the running notebooks
* **notebook** - Notebook files to be run
* **notebook.src** - Exported python source files exported from the notebooks
* **output** - Final output of the notebook computations
* **resultados** - Final results and analisys.

All the folders above are referred on the notebooks source code and can be
changed there. The values above are only the default values.

## Conclusion

This is a work in progress, but we already obtained good results in detecting
the relevant variables that can be used in order to optimize the collection
process.

**This repository contains only code as the data to be used on the notebooks is
confidential and contains personal information of customers.**


