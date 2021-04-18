################################33
# 统计推断
import os 
os.chdir(r'C:\Users\Administrator\Desktop\代码\源代码\Python_book\6Inference')

import pandas as pd
house_price_gr = pd.read_csv('house_price_gr.csv', encoding='gbk')
house_price_gr.head()

# 查看增长率分布是否符合正态分布
# %matplotlib inline
import seaborn as sns
from scipy import stats

sns.distplot(house_price_gr.rate, kde = True, fit=stats.norm)

# 使用qqplot看变量与正态分布接近程度
import statsmodels.api as sm 
from matplotlib import pyplot as plt 

fig = sm.qqplot(house_price_gr.rate, fit = True, line = '45')
fig.show()

# 区间估计
se = house_price_gr.rate.std()/len(house_price_gr) ** 0.5
LB = house_price_gr.rate.mean() - 1.98 * se 
UB = house_price_gr.rate.mean() + 1.98 * se
(LB, UB)

# 计算不同置信度下的置信区间
def confint(x, alpha = 0.05):
    n = len(x)
    xb = x.mean()
    df = n-1
    tmp = (x.std() / n **0.5) * stats.t.ppf(1-alpha/2, df)
    return {'Mean' : xb, 'Degree of freedom': df, 'LB':xb-tmp, 'UB':xb+tmp}

confint(house_price_gr.rate, 0.05)
confint(house_price_gr.rate, 0.01)

# 单样本t检验: 样本与总体的检验
d1 = sm.stats.DescrStatsW(house_price_gr.rate)
print('t-statistic = %6.4f, p-value = %6.4f, df=%s' %d1.ttest_mean(0.1))

#################################
#  双样本t检验: 两个样本之间的检验
# 数据源是信用卡
creditcard_exp= pd.read_csv(r'creditcard_exp.csv', skipinitialspace=True)
creditcard_exp = creditcard_exp.dropna(how='any')
creditcard_exp.head()

# 查看按性别分类后的描述性分析
creditcard_exp['avg_exp'].groupby(creditcard_exp['gender']).describe()

# 双样本t检验前，有三个基本条件
# 观测之间独立，两组服从正态分布，两组样本方差是否相同
# 故需进行方差齐次性检验
gender0 = creditcard_exp[creditcard_exp['gender']==0]['avg_exp']
gender1 = creditcard_exp[creditcard_exp['gender'] == 1]['avg_exp']
leveneTestRes = stats.levene(gender0, gender1, center = 'median')
print('w-value = %6.4f, p-value = %6.4f'%leveneTestRes)

# 方差齐，开始进行双样本t检验
stats.stats.ttest_ind(gender0, gender1, equal_var=True)

########################################
# 方差分析：检验多个样本均值是否有显著差异，或多于两个分类的分类变量与连续变量关系
# 单因素方差分析：检验一个分类变量与一个连续变量关系，其前提与双样本t检验相似
edu = []
for i in range(4):
    edu.append(creditcard_exp[creditcard_exp['edu_class'] == i]['avg_exp'])
stats.f_oneway(*edu)

# 多因素方差分析: 检验多个分类变量与一个连续变量的关系
# 还需考虑不同分类变量间的交互效应
# 可通过构建线性回归模型进行方差分析
# 注意：这里的C(edu_class)将变量中的每个类都进行了划分，如不加，则edu_class的value表示数值
from statsmodels.formula.api import ols
ana = ols('avg_exp ~ C(edu_class) + C(gender)', data = creditcard_exp).fit()
sm.stats.anova_lm(ana)
ana.summary()

# 考虑交互效应
ana1 = ols('avg_exp~C(edu_class) + C(gender) + C(edu_class)*C(gender)',\
     data = creditcard_exp).fit()
sm.stats.anova_lm(ana1)
ana1.summary()

#####################################33
# 相关分析（两连续变量关系检验）
# pearson可检测是否存在线性相关关系
# kendall检测两连续变量是否有非线性关系
creditcard_exp[['Income', 'avg_exp']].corr(method='pearson')
creditcard_exp.plot(x='Income', y = 'avg_exp', kind='scatter')

# 散点矩阵图：多个变量间相关关系进行直观全面了解
sns.pairplot(creditcard_exp[['avg_exp', 'Age', 'Income', 'dist_home_val', \
    'dist_avg_income']])
plt.show()

# 通过参数hue可以指定分组变量
sns.pairplot(creditcard_exp[['avg_exp', 'Age', 'Income', 'dist_home_val',\
    'dist_avg_income', 'gender']], 
    hue = 'gender', kind = 'reg', diag_kind = 'kde', size =1.5)
plt.show()

#######################################33
# 卡方检验（二分类变量关系检验）
# 先构建列联表来统计变量不同分类的频率，再通过卡方来进行检验
accepts = pd.read_csv('accepts.csv')
cross_table = pd.crosstab(accepts.bankruptcy_ind, columns = accepts.bad_ind,\
    margins = True)
cross_table

# 将频数转换为频率
cross_table_rowpct = cross_table.div(cross_table['All'], axis = 0)
cross_table_rowpct

# 再用卡方检验比较期望频数与实际频数的吻合程度
print('chisq = %6.4f\n p-value = %6.4f\n dof = %i\n expected_fre = %s'\
    %stats.chi2_contingency(cross_table))








