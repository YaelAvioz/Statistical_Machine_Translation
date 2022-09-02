###### **How To Run The Code:**

**Model 1:**

python get_alignment.py

There is option in the code to load the prob file that contains the prob table, instead to run the training phase for 15 iterations
I attached the file "tef_15_iter_model_1.npy".
The file is located in the given folder "results_model_1".
In order to use it the file should located in 'results_model1/tef_15_iter_model_1.npy' relatively to the working directory.
Otherwise, move or rename the probs file to a different location or name.

**Model 2:**

python get_alignment_model2.py

the files that needed to run this model are:
get_alignment_model2.py
test.py
train.py
data_utils.py

same as above (model 1) I attached file that contains the prob table
In order to use it the file should located in 'results_model2/tef_iter14_model2.npy' relatively to the working directory.
And also n_iter should be 1.