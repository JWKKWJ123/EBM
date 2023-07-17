import pandas as pd
from sklearn.model_selection import train_test_split
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.perf import ROC
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,roc_curve, classification_report,confusion_matrix,precision_score,roc_auc_score, auc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt



#function for bootstraping
n_iterations = 100
def bootstrap(a, b, calculate_statistic,name):
    # run bootstrap
    stats = list()
    for i in range(n_iterations):
        # prepare sample
        sample_a, sample_b = resample(a, b, stratify=a,  random_state=i)
        
        stat = calculate_statistic(sample_a, sample_b)
        stats.append(stat)
    average = np.average(stats)
    print(len(a),len(sample_a),str(calculate_statistic))
    metric = pd.DataFrame(stats)    
    metric.to_csv(" {}.csv".format(name))
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    return [average,lower, upper]
    
    
    
    
    
#load data
label = pd.read_csv(".csv")
data = pd.read_csv(".csv")

X_train,y_train = data,label['diagnosis']


label = pd.read_csv(" .csv")
data = pd.read_csv(" .csv")

X_test,y_test = data,label['diagnosis']

train_size = len(y_train)
test_size = len(y_test)
acc_list = []
sen_list = []
spe_list = []






#set model, include two pairwise biomarkers
ebm = ExplainableBoostingClassifier(interactions = 2)
#other hyperparameters
#ebm = ExplainableBoostingClassifier(interactions = 2, learning_rate=0.05, max_bins=256, min_samples_leaf=4, outer_bags=32)
#model training
ebm.fit(X_train,y_train)
 
 
output_pro = ebm.predict_proba(X_test)
output = ebm.predict(X_test)

ci_acc = bootstrap(y_test,output , accuracy_score,'V_acc_ADNI') 
ci_re = bootstrap(y_test,output , recall_score,'re_ADNI') 
ci_pre = bootstrap(y_test,output , precision_score,'pre_ADNI') 
ci_auc = bootstrap(y_test,output_pro[:,1] , roc_auc_score,'V_auc_ADNI') 

matrix=confusion_matrix(y_test,output)
sensitivity = float(matrix[1][1])/np.sum(matrix[1])
specificity = float(matrix[0][0])/np.sum(matrix[0][0]+matrix[0][1])

print(ci_acc)
print(ci_re)
print(ci_pre)
print(ci_auc)

    
### plot future importance
importance_list = np.zeros([13])
importances = ebm.term_importances("min_max")
names = ebm.term_names_


list = []
k=0
for (term, importance) in zip(names, importances):
    list.append([term,importance])
    k=k+1

list = pd.DataFrame(list)
list= list.rename(columns={list.columns[0]:'Feature'})
list= list.rename(columns={list.columns[1]:'Importance'})

list=list.sort_values('Importance',ascending=False)

f, ax = plt.subplots(figsize=(5, 10))

# Plot the feat importance
sns.set_color_codes("muted")
sns.barplot(x="Importance", y="Feature", data=list, label="EBM", color="orange", alpha=0.9, errorbar=('ci', 85))

# Add a legend and informative axis label
ax.legend(ncol=1, loc="lower right", frameon=True)
ax.set(xlim=(0, 2), ylabel="",
       xlabel="Mean Absolute Score")
sns.despine(left=True, bottom=True)
plt.savefig('  .png', dpi=800)
    
    
    
    
    
    
    
    