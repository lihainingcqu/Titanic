import numpy as np
import pandas as pd
# 定义函数：从姓名中获取头衔
def getTitle(name):
    str1=name.split( ',' )[1] #Mr. Owen Harris
    str2=str1.split( '.' )[0]#Mr
    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3
train = pd.read_csv("./train1.csv")
test = pd.read_csv("./test1.csv")
print ('训练数据集：',train.shape,'测试数据集：',test.shape)
rowNum_train=train.shape[0]
rowNum_test=test.shape[0]
print('kaggle训练数据集有多少行数据：',rowNum_train,
     ',kaggle测试数据集有多少行数据：',rowNum_test,)
full = train.append( test , ignore_index = True )
print ('合并后的数据集:',full.shape)
full.info()
# print('处理前：')
#年龄(Age)
full['Age']=full['Age'].fillna( full['Age'].mean() )

full.loc[full['Age'] <= 15, 'Age'] = 0
full.loc[(full['Age'] > 15) & (full['Age'] <= 20), 'Age'] = 1
full.loc[(full['Age'] > 20) & (full['Age'] <= 40), 'Age'] = 2
full.loc[(full['Age'] > 40) & (full['Age'] <= 60), 'Age'] = 3
# full.loc[(full['Age'] > 50) & (full['Age'] <= 60), 'Age'] = 4
full.loc[full['Age'] > 60, 'Age'] = 4
full.loc[full['Age'].isnull(), 'Age'] = 6
full['Age'] = full['Age'].astype(int)

#船票价格(Fare)
full['Fare'] = full['Fare'].fillna( full['Fare'].mean() )
# print('处理后：')
# full.info()
full['Embarked'].head()
full['Embarked'].value_counts()
# print(full['Embarked'].value_counts())
full['Embarked'] = full['Embarked'].fillna( 'S' )
#船舱号（Cabin）：查看里面数据长啥样
full['Cabin'].head()
#缺失数据比较多，船舱号（Cabin）缺失值填充为U，表示未知（Uknow）
full['Cabin'] = full['Cabin'].fillna( 'U' )
#检查数据处理是否正常

#查看性别数据这一列
full['Sex'].head()
# '''
# 将性别的值映射为数值
# 男（male）对应数值1，女（female）对应数值0
# '''
sex_mapDict={'male':1,
            'female':0}
#map函数：对Series每个数据应用自定义的函数计算
full['Sex']=full['Sex'].map(sex_mapDict)

# 登船港口(Embarked)

#存放提取后的特征
embarkedDf = pd.DataFrame()

# 使用get_dummies进行one-hot编码，产生虚拟变量（dummy variables），列名前缀是Embarked
embarkedDf = pd.get_dummies( full['Embarked'] , prefix='Embarked' )
embarkedDf.head()
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,embarkedDf],axis=1)

# 因为已经使用登船港口(Embarked)进行了one-hot编码产生了它的虚拟变量（dummy variables）
# 所以这里把登船港口(Embarked)删掉
full.drop('Embarked',axis=1,inplace=True)
full.head()
# 客舱等级(Pclass):
# 1=1等舱，2=2等舱，3=3等舱
#存放提取后的特征
pclassDf = pd.DataFrame()

pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
pclassDf.head()
full = pd.concat([full,pclassDf],axis=1)

#删掉客舱等级（Pclass）这一列
full.drop('Pclass',axis=1,inplace=True)
titleDf = pd.DataFrame()
titleDf['Title'] = full['Name'].map(getTitle)
# 乘客头衔每个名字当中都包含了具体的称谓，将这部分信息提取出来后可以作为非常有用一个新变量，可以帮助我们进行预测。

title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
titleDf['Age']=full['Age']

titleDf = pd.get_dummies(titleDf['Title'])
# print(titleDf.Mr.index(1))
titleDf.head()
# print(titleDf.head())
sum = lambda a,b: a + b
#存放客舱号信息
cabinDf = pd.DataFrame()
# 客场号的类别值是首字母，例如：C85 类别映射为首字母C
full[ 'Cabin' ] = full[ 'Cabin' ].map( lambda c : c[0] )

##使用get_dummies进行one-hot编码，列名前缀是Cabin
cabinDf = pd.get_dummies( full['Cabin'] , prefix = 'Cabin' )
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,cabinDf],axis=1)
#删掉客舱号这一列
full.drop('Cabin',axis=1,inplace=True)
#存放家庭信息
familyDf = pd.DataFrame()

'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
（因为乘客自己也是家庭成员的一个，所以这里加1）
'''
familyDf[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
# 家庭类别：
# 小家庭Family_Single：家庭人数=1
# 中等家庭Family_Small: 2<=家庭人数<=4
# 大家庭Family_Large: 家庭人数>=5

#if 条件为真的时候返回if前面内容，否则返回0
familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
familyDf[ 'Family_Small' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,familyDf],axis=1)
#相关性矩阵
corrDf = full.corr()

corrDf['Survived'].sort_values(ascending =False)
print(corrDf['Survived'].sort_values(ascending =False))
full_X = pd.concat( [titleDf,#头衔
                     pclassDf,#客舱等级
                     familyDf,#家庭大小
                     full['Fare'],#船票价格
                     cabinDf,#船舱号
                     embarkedDf,#登船港口
                     full['Sex'],#性别
                     full['Age']  # full['Age']
                    ] , axis=1 )
full_X.head()
#原始数据集有891行
sourceRow=891

# sourceRow是我们在最开始合并数据前知道的，原始数据集有总共有891条数据
# 从特征集合full_X中提取原始数据集提取前891行数据时，我们要减去1，因为行号是从0开始的。
#原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']

#预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]

from sklearn.cross_validation import train_test_split

#建立模型用的训练数据集和测试数据集
train_X, test_X, train_y, test_y = train_test_split(source_X ,
                                                    source_y,
                                                    train_size=.8)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
model = LogisticRegression()

# 随机森林Random Forests Model
from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100)

#支持向量机Support Vector Machines
# from sklearn.svm import SVC, LinearSVC
# model = SVC()

#Gradient Boosting Classifier
# from sklearn.ensemble import GradientBoostingClassifier
# model = GradientBoostingClassifier()

#K-nearest neighbors
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors = 3)

# Gaussian Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()

# scores = cross_val_score(model, source_X, source_y, cv=5)  #cv为迭代次数。
# print(scores)  # 打印输出每次迭代的度量值（准确度）
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）
model.fit( train_X , train_y )
# # 分类问题，score得到的是模型的正确率
# model.score(test_X , test_y )
print(model.score(test_X , test_y ))
# print(scores)

#使用机器学习模型，对预测数据集中的生存情况进行预测
pred_Y = model.predict(pred_X)

pred_Y=pred_Y.astype(int)
#乘客id
passenger_id = full.loc[sourceRow:,'PassengerId']
#数据框：乘客id，预测生存情况的值
predDf = pd.DataFrame(
    { 'PassengerId': passenger_id ,
     'Survived': pred_Y } )
predDf.shape
predDf.head()
#保存结果
predDf.to_csv( 'titanic_preddata.csv' , index = False )
