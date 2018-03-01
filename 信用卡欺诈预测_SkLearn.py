
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("creditcard.csv")
data.head()

count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1,1))

data = data.drop(['Time','Amount'],axis=1)
data.head()


X = data.ix[:, data.columns != 'Class'] #ix是索引行索引，区别loc 和iloc的索引区别
y = data.ix[:, data.columns == 'Class']

# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
                                                           #在没有欺诈行为的人中间选取和欺诈的人一样都多的数量
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])#将两个索引连接在一起
# print("under_sample_induece is:",under_sample_indices)


# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]  #根据索引取出under_sample的数据，备用
# print("under_sample_data is:",under_sample_data.head(5))


X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
# print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
# print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
# print("Total number of transactions in resampled data: ", len(under_sample_data))



from sklearn.model_selection import train_test_split  #交叉验证

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0) #在所有样本中划分训练集和测试集

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))
#
# # Undersampled dataset 在under_sample中划分训练集和测试集
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))

#
# #Recall = TP/(TP+FN) True Positive ，False Nagitive（划分错的，原本是Positive，现在划分在了Nagitive
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report


def printing_Kfold_scores(x_train_data,y_train_data):
    # fold = KFold(n_splits=5).split(len(y_train_data))
    fold = KFold(len(y_train_data),5)  #0.18之前的用法，返回的是index

    # Different C parameters
    c_param_range = [0.01,0.1,1,10,100]   #阈值，值越小，取到数后阈值越大；相当于的阈值的倒数

    results_table = pd.DataFrame( columns = ['C_parameter','Mean recall score'])

    results_table['C_parameter'] = c_param_range
    # print("results_table:", results_table)

#     # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        for iteration, indices in enumerate(fold,start=1):  #多次的交叉验证 start=1指index从1开始计数，indices就是fold的内容
            # print("indices:",indices)
            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C = c_param, penalty = 'l1')         #C表示正则化系数，取倒数

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            # print("y_train_data.iloc[indices[0],:].values:",y_train_data.iloc[indices[0], :].values)
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())
            # print("y_train_data.iloc[indices[0],:].values.ravel():", y_train_data.iloc[indices[0], :])
            # print("x_train_data.iloc[indices[0],:]:",x_train_data.iloc[indices[0],:])

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample) #掺入真实的label,和预测的label
            recall_accs.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')
    # return indirect statistics like the index value where the maximum values are attained
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')

    return best_c

best_c = printing_Kfold_scores(X_train_undersample,y_train_undersample)
plt.show()
#
# def plot_confusion_matrix(cm, classes,                          #混淆矩阵检查分类效果
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)  #cm: 要绘制的图像或数组。cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=0) # xticks( arange(12), calendar.month_name[1:13], rotation=17 )
#     plt.yticks(tick_marks, classes)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
import itertools
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
print("cnf_matrix:",cnf_matrix)
np.set_printoptions(precision=2)  #使用set_printoptions设置输出精度,保留两位小数

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))


def plot_confusion_matrix(cm, classes,                          #混淆矩阵检查分类效果
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  #cm: 要绘制的图像或数组。cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0) # xticks( arange(12), calendar.month_name[1:13], rotation=17 )
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):  #笛卡尔积：itertools.product(*iterables[, repeat])
        # print("i is :",i)
        # print("j is :",j)
        # print("cm.shape",cm.shape[0],cm.shape[1])
        plt.text(j, i, cm[i, j],           #cm[i,j]对应的数值填写在图面上
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()  #你的轴标签或标题（有时甚至是刻度标签）会超出图形区域，因此被截断。通过它自动调整调整，
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot non-normalized confusion matrix
class_names = [0,1]   #label的取值
plt.figure()
plot_confusion_matrix(cnf_matrix           #cm就是混淆矩阵的数组
                      , classes=class_names
                      , title='Confusion matrix')



#
best_c = printing_Kfold_scores(X_train,y_train)  #原始数据直接运行


#
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train,y_train.values.ravel())
y_pred_undersample = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred_undersample)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

#
lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)  #预测欺诈的可能性

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.figure(figsize=(10,10))

j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:,1] > i

    plt.subplot(3,3,j)
    j += 1

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i)
#
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
#
#
# credit_cards=pd.read_csv('creditcard.csv')
#
# columns=credit_cards.columns
# # The labels are in the last column ('Class'). Simply remove it to obtain features columns
# features_columns=columns.delete(len(columns)-1)
#
# features=credit_cards[features_columns]
# labels=credit_cards['Class']
#
#
#
# features_train, features_test, labels_train, labels_test = train_test_split(features,
#                                                                             labels,
#                                                                             test_size=0.2,
#                                                                             random_state=0)
#
#
# oversampler=SMOTE(random_state=0)
# os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
#
# len(os_labels[os_labels==1])
#
# os_features = pd.DataFrame(os_features)
# os_labels = pd.DataFrame(os_labels)
# best_c = printing_Kfold_scores(os_features,os_labels)
#
#
# lr = LogisticRegression(C = best_c, penalty = 'l1')
# lr.fit(os_features,os_labels.values.ravel())
# y_pred = lr.predict(features_test.values)
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(labels_test,y_pred)
# np.set_printoptions(precision=2)
#
# print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
#
# # Plot non-normalized confusion matrix
# class_names = [0,1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix
#                       , classes=class_names
#                       , title='Confusion matrix')
# plt.show()
#
