Code of Instance Specific Discriminative Modal Pursuit: A Serialized Approach.
For any question, please contact Yang Yang (yangy@lamda.nju.edu.cn); Ying Fan (fany@lamda.nju.edu.cn) or De-Chuan Zhan (zhandc@lamda.nju.edu.cn).
Enjoy the code.

**************************** Requirement ****************************
#requirement theano, pyflann

******************************* USAGE *******************************
imdb.py ----- The dataprocess of DMS (implementation)
dms.py  ----- The main code of the algorithm. It takes training data/label and parameters as input.

--the data file should contain:
#'train': train data set
#'test': test data set
#'trainlabel': train data label
#'testlabel': test data label
#'flens': the feature number of each modal

--the parameters
In the function of train_lstm:
#'modal_costs': the cost for each modal
#'max_costs': the cost budget that can been used in one specific scene

In the main:
#'recyl_maxlen': the max madalities that can been used

--demo:
data/: include the .mat for dataset of landsat, there is 4 modals which represent spectral values for 9 pixels
i.e.
python dms.py 2

***************************** REFERENCE *****************************
If you use this code in scientific work, please cite:
Yang Yang, De-Chuan Zhan, Ying Fan, and Yuan Jiang. Instance Specific Discriminative Modal Pursuit: A Serialized Approach. In: Proceedings of the 9th Asian Conference on Machine Learning (ACML'17) , Seoul, Korea, 2017.
*********************************************************************
