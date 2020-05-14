import pandas as pd
from surprise import Dataset
from surprise import BaselineOnly
from surprise import Reader
import sys

input_path = sys.argv[1] + "yelp_train.csv"
test_file_name = sys.argv[2]
result_file_name = sys.argv[3]

reader = Reader(rating_scale=(0, 5))
train_read = pd.read_csv(input_path)
test_read = pd.read_csv(test_file_name)

train_load = Dataset.load_from_df(train_read, reader=reader).build_full_trainset()
test_load = Dataset.load_from_df(test_read, reader=reader).build_full_trainset()

bsl_options = {'method': 'als', 'n_epochs': 9, 'reg_u': 7.6, 'reg_i': 3.5}

algorithm = BaselineOnly(bsl_options=bsl_options)
results = algorithm.fit(train_load).test(test_load.build_testset())

with open(result_file_name, "w") as fout:
    fout.write("user_id, business_id, prediction\n")
    for p in results:
        fout.write(str(p.uid) + "," + str(p.iid) + "," + str(p.est) + "\n")
fout.close()
