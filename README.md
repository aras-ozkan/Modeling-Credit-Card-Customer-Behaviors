In this project my task was to predict 6 different targets. Each of them is binary classification. However, most of the labels were missing. So, in order to correctly train my learners, I divided this task into 6. Each of the tasks handling according target. I preprocessed X data by imputing most frequent one for object data-typed columns (one hot encoded these since they can’t be given as learner algorithm inputs) and mean for numerical columns. I also dropped columns that have more than 40% missing data because it has no validity if more than 40% is missing. In order to acquire my labels, I used non-missing labels to train Random Forest and K-NN learners and used LogisticRegression to Ensemble (Each target has separate learners from all these three types). After training this system, I noticed that for both precision and AUROC Random Forest was the most successful algorithm for this task. So, I used Random Forest to predict test results. Below are my training process results tested with 20% of training data that I did not use for training. The data cannot be uploaded to Git Hub since the file is too large. I worked with over 120,000 data points each consisting 200 attributes. 

#Results

#RANDOM-FOREST
Target 1:
Score:  0.7730279898218829
[[2952  105]
[ 787   86]]
AUROC:  0.6397105248465487
Target 2:
Score:  0.8077803203661327
[[1726   27]
[ 393   39]]
AUROC:  0.6916661384716148
Target 3:
Score:  0.7951007910181168
[[3001  124]
[ 679  115]]
AUROC:  0.7195645340050377
Target 4:
Score:  0.8138037599793974
[[3020  140]
[ 583  140]]
AUROC:  0.7590936148607245
Target 5:
Score:  0.7961783439490446
[[3057   96]
[ 704   68]]
AUROC:  0.6751446931863561
Target 6:
Score:  0.8078680203045685
[[3024  116]
[ 641  159]]
AUROC:  0.7349341162420382

#KNNs
Target 1:
Score:  0.7412213740458016
[[2819  238]
[ 779   94]]
AUROC:  0.5311880681709602
Target 2:
Score:  0.7762013729977116
[[1640  113]
[ 376   56]]
AUROC:  0.5867798060467769
Target 3:
Score:  0.7629497320745088
[[2900  225]
[ 704   90]]
AUROC:  0.5886289168765744
Target 4:
Score:  0.7744012361576101
[[2919  241]
[ 635   88]]
AUROC:  0.6187439816516973
Target 5:
Score:  0.7745222929936306
[[2964  189]
[ 696   76]]
AUROC:  0.5543158173234144
Target 6:
Score:  0.7804568527918782
[[2980  160]
[ 705   95]]
AUROC:  0.5973765923566879

#LOGISTIC-REGRESSION
Target 1:
Score:  0.7730279898218829
[[2952  105]
[ 787   86]]
AUROC:  0.5450802825730742
Target 2:
Score:  0.8077803203661327
[[1726   27]
[ 393   39]]
AUROC:  0.5641281612473854
Target 3:
Score:  0.7951007910181168
[[3001  124]
[ 679  115]]
AUROC:  0.5678172292191437
Target 4:
Score:  0.8138037599793974
[[3020  140]
[ 583  140]]
AUROC:  0.589327827091759
Target 5:
Score:  0.7961783439490446
[[3057   96]
[ 704   68]]
AUROC:  0.5449571836346336
Target 6:
Score:  0.8078680203045685
[[3024  116]
[ 641  159]]
AUROC:  0.599625398089172


For the data contact: arasozkan576@gmail.com
