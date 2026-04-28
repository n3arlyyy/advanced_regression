#подключаем необходимые библиотеки
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as seab
color = seab.color_palette()
seab.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew

#загрузка тестовых и тренировочных данных

train = pd.read_csv('train (1).csv')
test = pd.read_csv('test (1).csv')

df = pd.DataFrame(train)
print(f"all: {len(df.columns)}")
numeric_cols = df.select_dtypes(include=[np.number])
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool'])
print(f"numerical: {len(numeric_cols.columns)}")
print(f"categorial: {len(categorical_cols.columns)}")

#удаление столбца id, так как он не нужен для обучения
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))

#график зависимости цены от площади
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
###plt.show()

#удаление выбросов
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#проверка графика после удаления выбросов
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
###plt.show()

seab.distplot(train['SalePrice'] , fit=norm);

#поиск среднего значения и стандартного отклонения
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#график для распределения
plt.legend([rf'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f})'], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#квантиль-квантильный график
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
###plt.show()

#приближение к нормальному распределению
train["SalePrice"] = np.log1p(train["SalePrice"])

#проверка распределения 
seab.distplot(train['SalePrice'] , fit=norm);

#поиск среднего значения и стандартного отклонения
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#график для распределения
plt.legend([rf'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f})'], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#квантиль-квантильный график
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
###plt.show()

#объединение тестовых и тренировочных данных
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

#пропущенные значения
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(20))

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='vertical')
seab.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
###plt.show()

df = pd.DataFrame(train)
print(f"all: {len(df.columns)}")
numeric_cols = df.select_dtypes(include=[np.number])
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool'])
print(f"numerical: {len(numeric_cols.columns)}")
print(f"categorial: {len(categorical_cols.columns)}")

#корелляционная карта для цены
corrmat = train.corr(numeric_only=True)
plt.subplots(figsize=(12,9))
seab.heatmap(corrmat, vmax=0.9, square=True)
###plt.show()

#ввод прорпущенных значений
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#группировка по районам и заполнение медианными для района значениями
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#проверка, остались ли пропущенные значения
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head())

#преобразование нужных числовых столбцов в категориальные
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#год и месяц продажи становятся категориальными
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

df = pd.DataFrame(train)
print(f"all: {len(df.columns)}")
numeric_cols = df.select_dtypes(include=[np.number])
n_cols = df.select_dtypes(include=['object'])
o_cols = df.select_dtypes(include=['category'])
categorical_cols = df.select_dtypes(include=['bool'])
print(f"numerical: {len(numeric_cols.columns)}")
print(f"object: {len(n_cols.columns)}")
print(f"categorial: {len(o_cols.columns)}")
print(f"bool: {len(categorical_cols.columns)}")
print('--------')
print(n_cols.columns)

from sklearn.preprocessing import LabelEncoder
cols = ('MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition')

#обработка столбцов и применение LabelEncoder к категориальным признакам
for c in cols:
    lbl = LabelEncoder() 
    all_data[c] = lbl.fit_transform(list(all_data[c].values))

#размер данных       
print('Shape all_data: {}'.format(all_data.shape))

df = pd.DataFrame(train)
print(f"all: {len(df.columns)}")
numeric_cols = df.select_dtypes(include=[np.number])
n_cols = df.select_dtypes(include=['object'])
o_cols = df.select_dtypes(include=['category'])
categorical_cols = df.select_dtypes(include=['bool'])
print(f"numerical: {len(numeric_cols.columns)}")
print(f"object: {len(n_cols.columns)}")
print(f"categorial: {len(o_cols.columns)}")
print(f"bool: {len(categorical_cols.columns)}")
print('--------')
print(n_cols.columns)

#добавление общей площади
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

df = pd.DataFrame(train)
print(f"all: {len(df.columns)}")
numeric_cols = df.select_dtypes(include=[np.number])
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool'])
print(f"numerical: {len(numeric_cols.columns)}")
print(f"categorial: {len(categorical_cols.columns)}")

#проверка скошенности
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

df = pd.DataFrame(train)
print(f"all: {len(df.columns)}")
numeric_cols = df.select_dtypes(include=[np.number])
n_cols = df.select_dtypes(include=['object'])
o_cols = df.select_dtypes(include=['category'])
categorical_cols = df.select_dtypes(include=['bool'])
print(f"numerical: {len(numeric_cols.columns)}")
print(f"object: {len(n_cols.columns)}")
print(f"categorial: {len(o_cols.columns)}")
print(f"bool: {len(categorical_cols.columns)}")
print('--------')
print(n_cols.columns)

#финальное преобразование
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
all_data[skewed_features] = np.log1p(all_data[skewed_features])
all_data = pd.get_dummies(all_data)
print(all_data.shape)

#получение новых тестовых и тренировочных данных
train = all_data[:ntrain]
test = all_data[ntrain:]

df = pd.DataFrame(train)
print(f"all: {len(df.columns)}")
numeric_cols = df.select_dtypes(include=[np.number])
n_cols = df.select_dtypes(include=['object'])
o_cols = df.select_dtypes(include=['category'])
categorical_cols = df.select_dtypes(include=['bool'])
print(f"numerical: {len(numeric_cols.columns)}")
print(f"object: {len(n_cols.columns)}")
print(f"categorial: {len(o_cols.columns)}")
print(f"bool: {len(categorical_cols.columns)}")
print('--------')
print(n_cols.columns)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

#функция валидации
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    #клонируем исходные модели
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        #тренируем клонированные модели и создаем нестандартные прогнозы для обучения метамодели
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        #тренируем метамодель, используя нестандартные прогнозы
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #создание прогнозов всех базовых моделей на основе тестовых данных и использование усредненных прогнозов
    #в качестве мета-признаков для окончательного прогноза, который выполняется метамоделью
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.80 +
               xgb_train_pred*0.20))

ensemble = stacked_pred*0.80 + xgb_pred*0.20

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv(r'C:\Users\ivo-1\Desktop\tot_submission.csv', index=False)
