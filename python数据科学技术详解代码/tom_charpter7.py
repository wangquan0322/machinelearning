##############################################
# 一元线性回归
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels import formula
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

os.chdir(r"C:\Users\Administrator\Desktop\machine_learning\python数据科学技术详解代码\rawdata")


raw = pd.read_csv(r'creditcard_exp.csv', skipinitialspace=True)
raw.head()

# 数据清洗
exp = raw[raw['avg_exp'].notnull()].copy().iloc[:, 2:].drop('age2',axis=1)

exp_new = raw[raw['avg_exp'].isnull()].copy().iloc[:, 2:].drop('age2',axis=1)

exp.describe(include='all')

# 确认变量是否有线性关系
exp[['Income', 'avg_exp', 'Age', 'dist_home_val']].corr(method = 'pearson')

# avg_exp与income存在较高相关性，故可用简单线性回归
lm_s = ols('avg_exp ~ Income', data = exp).fit()
#ols模型训练后的参数
print(lm_s.params)
#模型输出线性结果
lm_s.summary() 

# 根据模型给出y的预测值和残差
pd.DataFrame([lm_s.predict(exp), lm_s.resid], index = ['predict', 'resid']).T.head()

# 同样用模型使用在待预测数据集上
lm_s.predict(exp_new)[:5]

##########################################################
# 多元线性回归
lm_m = ols('avg_exp ~ Age+Income+dist_home_val+dist_avg_income', data = exp).fit()
lm_m.summary()


# 前向回归法（筛选变量）
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response, ' + '.join(selected + [candidate])
            )
            aic = ols(formula=formula, data = data).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = aic_with_candidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print('aic is {}, continuing!'.format(current_score))
        else:
            print('forward selection over')
            break
    formula = "{} ~ {}".format(response, '+'.join(selected))
    print('final formula is {}'.format(formula))
    model = ols(formula = formula, data = data).fit()
    return model

# 测试集测试前向回归法
data_for_select = exp[['avg_exp', 'Income', 'Age', 'dist_home_val', 'dist_avg_income']]
lm_m = forward_select(data=data_for_select, response = 'avg_exp')
print(lm_m.rsquared)

########################################
# 残差分析
anal = lm_s

exp['Pred'] = anal.predict(exp)
exp['resid'] = anal.resid
exp.plot('Income', 'resid', kind = 'scatter') # 画出残差与自变量的散点图

# 对于残差的异方差性，可对因变量avg_exp取对数并重新建模
ana2 = ols('avg_exp_ln ~ Income', exp).fit()
exp['Pred'] = ana2.predict(exp)
exp['resid'] = ana2.resid
exp.plot('Income', 'resid', kind ='scatter')
plt.show()

# 从业务上，也可以同时对自变量和因变量取对数
# 表示自变量增加多少百分点，因变量也增加了多少百分点
exp['Income_ln'] = np.log(exp['Income'])
ana3 = ols('avg_exp_ln ~ Income_ln', exp).fit()
exp['resid'] = ana3.predict(exp)
exp.plot('Income_ln', 'resid', kind = 'scatter')

# 比较不同情况下一元线性回归结果
r_sq = {'exp~Income': anal.rsquared, 'ln(exp) ~ Income':ana2.rsquared,\
    'ln(exp) ~ ln(Income)':ana3.rsquared}
print(r_sq)

# 线性回归异常值点可用学生化残差判定
exp['resid_t'] = (exp['resid'] - exp['resid'].mean()) / exp['resid'].std()
exp[abs(exp['resid_t']) > 2]

# 去除异常点后重新建模
exp2 = exp[abs(exp['resid_t']) <= 2].copy()
ana4 = ols('avg_exp_ln ~ Income_ln', exp2).fit()
exp2['Pred'] = ana4.predict(exp2)
exp2['resid'] = ana4.resid
exp2.plot('Income', 'resid', kind ='scatter')
plt.show()

# 通过statsmodels包一次性返回多个异常点判断的统计量
from statsmodels.stats.outliers_influence import OLSInfluence
OLSInfluence(ana3).summary_frame().head() 

# 多重共线性VIF
exp2['dist_home_val_ln'] = np.log(exp2['dist_home_val'])
exp2['dist_avg_income_ln'] = np.log(exp2['dist_avg_income'])

# 定义VIF函数
def vif(df, col_i):
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i +'~'+'+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1./(1. - r2)

# 计算VIF
exog = exp2[['Age','Income_ln','dist_home_val_ln','dist_avg_income_ln']]
for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i = i))

# 对多重共线的变量可以删除一个，或者做其他处理，如相除
exp2['high_avg_ratio'] = exp2['high_avg']/exp2['dist_avg_income']

#再次判断
exog1 = exp2[['Age','high_avg_ratio','dist_home_val_ln','dist_avg_income_ln']]
for i in exog1.columns:
    print(i, '\t', vif(df=exog1, col_i = i))



#######################################################
# 正则化（岭回归和lasso回归）
# 正则化也可以减少变量，从而降低多重共线性的影响
# 岭回归-L2正则 lasso回归-L1正则
# 参数L1_wt =0是岭回归，=1是lasso回归
lmr = ols('avg_exp ~ Income + dist_home_val + dist_avg_income', \
    data = exp).fit_regularized(alpha = 1, L1_wt = 0)
lmr.summary()

# lasso回归
lmr1 = ols('avg_exp ~ Income + dist_home_val + dist_avg_income', \
    data = exp).fit_regularized(alpha = 1, L1_wt = 1)
lmr1.summary()

#由于statismodel的参数alpha不能连续输入，故使用sklearn来进行快速调参
#但是sklearn中的数据不会自动标准化，从而不同自变量可能有量纲不同的影响
#因此我们需要先将自变量进行标准化，从而避免量纲不同的影响
from sklearn.preprocessing import StandardScaler
continuous_xcols = ['Age', 'Income', 'dist_home_val', 
                    'dist_avg_income']   #  抽取连续变量
scaler = StandardScaler() #标准化
X = scaler.fit_transform(exp[continuous_xcols])
y = exp['avg_exp_ln']

# 使用不同正则化系数进行交叉验证
from sklearn.linear_model import RidgeCV
alphas = np.logspace(-2, 3, 100, base = 10)

# Search the min MSE by CV
rcv = RidgeCV(alphas=alphas, store_cv_values=True) 
rcv.fit(X, y)

# 输出结果
print('The best alpha is {}'.format(rcv.alpha_))
print('The r-square is {}'.format(rcv.score(X, y))) 

# 将正则化系数搜索空间的交叉验证结果进行可视化
cv_values = rcv.cv_values_
n_fold, n_alphas = cv_values.shape

cv_mean = cv_values.mean(axis=0)
cv_std = cv_values.std(axis=0)
ub = cv_mean + cv_std / np.sqrt(n_fold)
lb = cv_mean - cv_std / np.sqrt(n_fold)

plt.semilogx(alphas, cv_mean, label='mean_score')
plt.fill_between(alphas, lb, ub, alpha=0.2)
plt.xlabel("$\\alpha$")
plt.ylabel("mean squared errors")
plt.legend(loc="best")
plt.show()


# 手动选择正则化系数——根据业务判断
# 岭迹图可将不同正则化系数下的变量系数保存下来
from sklearn.linear_model import Ridge

ridge = Ridge()

coefs = []
for alpha in alphas:
    ridge.set_params(alpha=alpha)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# 使用正则化系数40，输出系数和结果
ridge.set_params(alpha=40)
ridge.fit(X, y)
ridge.coef_

ridge.score(X, y)
np.exp(ridge.predict(X)[:5])

# sklearn使用lasso
from sklearn.linear_model import LassoCV

lasso_alphas = np.logspace(-3, 0, 100, base=10)
lcv = LassoCV(alphas=lasso_alphas, cv=10) # Search the min MSE by CV
lcv.fit(X, y)

print('The best alpha is {}'.format(lcv.alpha_))
print('The r-square is {}'.format(lcv.score(X, y))) 

# 
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso_coefs = []
for alpha in lasso_alphas:
    lasso.set_params(alpha=alpha)
    lasso.fit(X, y)
    lasso_coefs.append(lasso.coef_)

# 画图
ax = plt.gca()

ax.plot(lasso_alphas, lasso_coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

lcv.coef_









