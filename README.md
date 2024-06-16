# Evaluation codes for Chat Autosuggest
## Pre-requisite: all outputs are in the ```./outputs``` folder in the format out.{type}.{dataset}.{model} and required types are seen and unseen for each (dataset, model) combination.

combine seen and unseen outputs
```sh
python code/utils/combine.py
```
make buckets on the prefix of outputs (type=all is used)
```sh
mkdir Poutputs
python code/pbucket/make_pbuckets.py
```
use analyze automate to run evaluation (code/eval/eval_n.py file) on outputs using multiprocessing
```sh
mkdir results
python analysis/analyze_automate.py --multi --cores 40
```
and also on prefix buckets
```sh
mkdir Presults
python analysis/analyze_automate.py --multi --cores 40 --pb
```
now we accumulate the results in a more readable format.
```sh
python analysis/accumulate_results.py --d ddc
python analysis/accumulate_results.py --d dstc7
python analysis/Paccumulate_results.py --d ddc
python analysis/Paccumulate_results.py --d dstc7

```
find the results at the csv files created. (like ```./ddc.csv``` and ```./Pddc.csv```)

also to create tables for each model across models, run
```sh
python analysis/make_table.py
```
which makes a folder named tables containing the tables.