#!/usr/bin/env python
# coding: utf-8

# # Prepare Problem

# ## Load libraries
# 
# Zona de importação das bibliotecas utilizadas neste trabalho.

# In[302]:


import random
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import scipy as sp
import scipy.stats as stats
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import BaggingRegressor
import pickle
import os
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# ## Load dataset
# 
# É utilizado a biblioteca "Pandas" para se proceder à importação do ficheiro "pmsm_temperature_data.csv".
# Neste ficheiro estão identificados vários dados de sensores adquiridos através de um "permanent magnet synchronous motor" (PMSM).  
# É feita uma primeira visualização de uma pequena amostra das primeiras linhas do dataframe, para ter uma ideia do conteúdo dos dados.

# In[303]:


electric_motor_temp = pd.read_csv('pmsm_temperature_data.csv')
electric_motor_temp.head()


# # Descriptive statistics, Data visualizations, Data Cleaning, Data transform
# 
# Nesta fase é feita a visualização das estatísticas descritivas e dos dados, e é realizado o tratamento e a limpeza dos mesmos.

# In[3]:


electric_motor_temp.info()


# Podemos observar que o dataframe tem 998070 dados, tem 13 colunas, todos os dados são do tipo numpy float64, à excepção dos dados da coluna "profile_id", que é do tipo numpy int64.

# In[4]:


electric_motor_temp.isnull().any()


# O dataframe não tem quaisqueres missing values, i.e., None ou NaN ou Null. Por esse motivo não é necessário proceder ao tratamento de missing values.

# In[5]:


electric_motor_temp.describe()


# Com esta informação, é permitido ter uma visão geral do dataframe.

# Existe a necessidade de visualizar as horas das sessões de medição de forma a descobrir outliers, isto porque as temperaturas dentro dos motores elétricos precisam de tempo para variar.

# In[6]:


fig = plt.figure(figsize=(17, 5))
grpd = electric_motor_temp.groupby(['profile_id'])
df = grpd.size().sort_values().rename('hours').reset_index()
ordered_ids = df.profile_id.values.tolist()
sns.barplot(y='hours', x='profile_id', data=df, order=ordered_ids)
tcks = plt.yticks(2*3600*np.arange(1, 8), [f'{h} hours' for h in range(1, 8)])


# Devido ao facto de termos quase 1 milhão de registos, decidi apenas incorporar os registos em que os "profile_id" contenham mais de 4 horas (28800 segundos = 4 horas), diminuindo assim o tempo de processamento e evitando erros de memória.

# In[304]:


id_hours = electric_motor_temp.groupby(['profile_id']).size().sort_values().rename('hours').reset_index()
list_ids = id_hours[id_hours.hours>28800].profile_id.tolist()
final_df = pd.DataFrame(columns=['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d', 'i_q', 'pm', 'stator_yoke', 'stator_winding', 'stator_tooth', 'profile_id'])
for id_profile in list_ids:
    df_tmp = electric_motor_temp[electric_motor_temp.profile_id==id_profile]
    final_df = final_df.append(df_tmp)
electric_motor_temp = final_df
electric_motor_temp.describe()


# Iremos passar a ter 359805 registos.

# A variável "profile_id" irá ser apagada por não ser útil para o nosso objetivo.

# In[305]:


del electric_motor_temp['profile_id']


# Iremos averiguar se existem outliers em cada uma das variáveis.
# São apagados todos os outliers de forma automática, caso existam.

# Função que devolve os limite inferior e superior do boxplot, de acordo com a variável que é passada.

# In[306]:


def get_whiskers(field):
    q1 = electric_motor_temp[field].quantile(0.25)
    q3 = electric_motor_temp[field].quantile(0.75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    return lower_whisker, upper_whisker


# In[10]:


sns.boxplot(electric_motor_temp['ambient'])


# In[307]:


lower_whisker, upper_whisker = get_whiskers('ambient')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['ambient'] < lower_whisker) | (electric_motor_temp['ambient'] > upper_whisker)].index,inplace=True)


# In[12]:


sns.boxplot(electric_motor_temp['coolant'])


# In[308]:


lower_whisker, upper_whisker = get_whiskers('coolant')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['coolant'] < lower_whisker) | (electric_motor_temp['coolant'] > upper_whisker)].index,inplace=True)


# In[14]:


sns.boxplot(electric_motor_temp['u_d'])


# In[309]:


lower_whisker, upper_whisker = get_whiskers('u_d')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['u_d'] < lower_whisker) | (electric_motor_temp['u_d'] > upper_whisker)].index,inplace=True)


# In[16]:


sns.boxplot(electric_motor_temp['u_q'])


# In[310]:


lower_whisker, upper_whisker = get_whiskers('u_q')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['u_q'] < lower_whisker) | (electric_motor_temp['u_q'] > upper_whisker)].index,inplace=True)


# In[18]:


sns.boxplot(electric_motor_temp['motor_speed'])


# In[311]:


lower_whisker, upper_whisker = get_whiskers('motor_speed')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['motor_speed'] < lower_whisker) | (electric_motor_temp['motor_speed'] > upper_whisker)].index,inplace=True)


# In[20]:


sns.boxplot(electric_motor_temp['torque'])


# In[312]:


lower_whisker, upper_whisker = get_whiskers('torque')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['torque'] < lower_whisker) | (electric_motor_temp['torque'] > upper_whisker)].index,inplace=True)


# In[22]:


sns.boxplot(electric_motor_temp['i_d'])


# In[313]:


lower_whisker, upper_whisker = get_whiskers('i_d')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['i_d'] < lower_whisker) | (electric_motor_temp['i_d'] > upper_whisker)].index,inplace=True)


# In[24]:


sns.boxplot(electric_motor_temp['i_q'])


# In[314]:


lower_whisker, upper_whisker = get_whiskers('i_q')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['i_q'] < lower_whisker) | (electric_motor_temp['i_q'] > upper_whisker)].index,inplace=True)


# In[26]:


sns.boxplot(electric_motor_temp['pm'])


# In[315]:


lower_whisker, upper_whisker = get_whiskers('pm')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['pm'] < lower_whisker) | (electric_motor_temp['pm'] > upper_whisker)].index,inplace=True)


# In[28]:


sns.boxplot(electric_motor_temp['stator_yoke'])


# In[316]:


lower_whisker, upper_whisker = get_whiskers('stator_yoke')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['stator_yoke'] < lower_whisker) | (electric_motor_temp['stator_yoke'] > upper_whisker)].index,inplace=True)


# In[30]:


sns.boxplot(electric_motor_temp['stator_tooth'])


# In[317]:


lower_whisker, upper_whisker = get_whiskers('stator_tooth')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['stator_tooth'] < lower_whisker) | (electric_motor_temp['stator_tooth'] > upper_whisker)].index,inplace=True)


# In[32]:


sns.boxplot(electric_motor_temp['stator_winding'])


# In[318]:


lower_whisker, upper_whisker = get_whiskers('stator_winding')
electric_motor_temp.drop(electric_motor_temp[(electric_motor_temp['stator_winding'] < lower_whisker) | (electric_motor_temp['stator_winding'] > upper_whisker)].index,inplace=True)


# In[319]:


electric_motor_temp.describe()


# Após a remoção dos outliers, ficamos com 289758 registos

# São criadas 2 novas variáveis e eliminadas as variáveis que ajudaram a criar essas novas variáveis.

# In[320]:


electric_motor_temp['voltage'] = np.sqrt(electric_motor_temp['u_d']**2 + electric_motor_temp['u_q']**2)
electric_motor_temp['current'] = np.sqrt(electric_motor_temp['i_d']**2 + electric_motor_temp['i_q']**2)
electric_motor_temp['apparent_power'] = electric_motor_temp['voltage'] * electric_motor_temp['current']
electric_motor_temp['effective_power'] = electric_motor_temp['i_d'] * electric_motor_temp['u_d'] + electric_motor_temp['i_q'] * electric_motor_temp['u_q']

del electric_motor_temp['u_d']
del electric_motor_temp['u_q']
del electric_motor_temp['i_q']
del electric_motor_temp['i_d']

electric_motor_temp.head()


# Vamos agora averiguar se existem duplicados.

# In[36]:


print("Existem {} duplicados".format(electric_motor_temp.duplicated().sum()))


# # Feature Selection
# 
# São aplicados vários métodos de feature selection para identificar as variáveis com maior importância para o nosso target "pm".

# In[321]:


features = electric_motor_temp.drop('pm', 1)
target = electric_motor_temp['pm']


# ## F-Test

# In[38]:


best_features=SelectKBest(score_func=f_regression,k=5).fit_transform(features,target)

print(best_features[:5])
print("")
print(features.head())


# Com a informação acima, conseguimos concluir através deste método que as variáveis "coolant", "motor_speed", "stator_yoke", "stator_tooth" e "stator_winding" são as 5 variáveis com maior importância.

# ## Heatmap

# Irá ser criado um heatmap para nos dar uma visualização de quais as variáveis com maior correlação entre elas e analisar as variáveis que são de maior interesse para o nosso estudo. Neste caso queremos saber quais as variáveis com maior correlação com a variável "pm".

# In[39]:


f, ax = plt.subplots(figsize = (15,15))
mask = np.zeros_like(electric_motor_temp.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(electric_motor_temp.corr(), mask=mask, annot=True, fmt=".3f", linewidths=0.5, ax=ax, cmap='RdYlGn')


# É possivel verificar que as variáveis que têm maior correlação com a variável "pm" são as variáveis "stator_yoke", "stator_winding" e "stator_tooth".

# ## Pairplot

# De seguida iremos criar um Pairplot com variáveis que têm maior correlação com a variável "pm", i.e., "stator_yoke", "stator_winding" e "stator_tooth".

# In[322]:


features = ['stator_yoke', 'stator_winding', 'stator_tooth', 'pm']
sns.pairplot(electric_motor_temp, vars=features, aspect=0.5)


# In[41]:


f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.distplot(electric_motor_temp["pm"], color="blue", ax=axes[0, 0])
sns.distplot(electric_motor_temp["stator_yoke"], color="red", ax=axes[0, 1])
sns.distplot(electric_motor_temp["stator_winding"], color="green", ax=axes[1, 0])
sns.distplot(electric_motor_temp["stator_tooth"], color="yellow", ax=axes[1, 1])


# Iremos passar à análise das variavéis com maior correlação.

# ## Regressão Linear
# 
# É realizada regressão linear entre as variáveis descritas.
# É calculado o coeficiente de determinação, correlação de pearson e spearman entre as mesmas.

# In[42]:


def linear_regression(field_1, field_2):
    var_1 = electric_motor_temp[field_1]
    var_2 = electric_motor_temp[field_2]
    
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(var_1, var_2)
    electric_motor_temp.plot(x=field_1, y=field_2, kind='scatter')
    plt.plot(var_1,var_1*slope+intercept,'r')
    plt.show()
    print ("r-squared : {}".format(r_value**2))
    print ("Pearson correlation : {}".format(pearsonr(var_1, var_2)[0]))
    print ("Spearman correlation : {}".format(spearmanr(var_1, var_2)[0]))


# In[43]:


linear_regression('pm', 'stator_yoke')


# In[44]:


linear_regression('pm', 'stator_winding')


# In[45]:


linear_regression('pm', 'stator_tooth')


# Apesar da quantidade de dados ser muito grande e dispares, conseguimos concluir que:
# 
# - O valor do campo "pm" aumenta significativamente com o aumento das variáveis "stator_yoke", "stator_winding" e "stator_tooth".

# ## Kolmogorov-Smirnov test, CDF e ECDF

# In[46]:


def distribution(field):
    var = electric_motor_temp[field]
    sns.kdeplot(var, shade=True)
    return


# In[47]:


def sk_plot_norm(field):
    var = electric_motor_temp[field]
    length = len(var)
    mu = sp.mean(var)
    plt.figure(figsize=(12, 7))
    plt.plot(np.sort(var), np.linspace(0, 1, length), linewidth=3.0)
    plt.plot(np.sort(stats.norm.rvs(loc=mu, scale=0.5, size=length)), np.linspace(0, 1, length), linewidth=3.0, color="r")
    plt.legend('top right')
    plt.legend(['CDF', 'ECDF'])
    plt.title(field)
    plt.show()


# In[48]:


def cdf(field):
    var = electric_motor_temp[field]
    var.hist(cumulative = True)
    sorted_field = np.sort(var)
    plt.step(sorted_field, np.arange(sorted_field.size), linewidth=5.0, color="r")
    plt.show()


# In[49]:


def sk_test_2samp(field):
    var = electric_motor_temp[field]
    length = len(var)
    half_length = int(length/2)
    first_half = var.iloc[half_length:]
    second_half = var.iloc[:half_length]
    p_value = 0.05
    sk_2samp = stats.ks_2samp(first_half, second_half)
    print (str(field.upper()) + "\n\n" + str(sk_2samp))
    if sk_2samp[1] < p_value:
        print("É rejeitada a hipótese nula")
    else:
        print("Não é rejeitada a hipótese nula")
        
    return first_half, second_half


# In[50]:


def ks_plot_comp_cdf(field, first_half, second_half):
    mu_1 = sp.mean(first_half)
    mu_2 = sp.mean(second_half)
    plt.figure(figsize=(12, 7))
    
    if mu_1 > mu_2:
        diff_mu = mu_1 - mu_2
        plt.plot(np.sort(first_half), np.linspace(0, 1, len(first_half), endpoint=False), linewidth=3.0)
        plt.plot(np.sort(second_half + diff_mu), np.linspace(0, 1, len(second_half), endpoint=False), linewidth=3.0, color="r")
    if mu_1 < mu_2:
        diff_mu = mu_2 - mu_1
        plt.plot(np.sort(first_half + diff_mu), np.linspace(0, 1, len(first_half), endpoint=False), linewidth=3.0)
        plt.plot(np.sort(second_half), np.linspace(0, 1, len(second_half), endpoint=False), linewidth=3.0, color="r")
        
    plt.legend('top right')
    plt.legend(['First half', 'Second half'])
    plt.title('Comparing ' + str(field) + ' first half and fecond half CDFs')
    plt.xticks([])
    plt.show()


# In[51]:


def ks_plot_comp_ecdf(field, first_half, second_half):
    mu_1 = sp.mean(first_half)
    mu_2 = sp.mean(second_half)
    plt.figure(figsize=(12, 7))
    
    if mu_1 > mu_2:
        diff_mu = mu_1 - mu_2
        plt.plot(np.sort(stats.norm.rvs(loc=first_half, scale=5, size=len(first_half))), np.linspace(0, 1, len(first_half), endpoint=False), linewidth=3.0, color="g")
        plt.plot(np.sort(stats.norm.rvs(loc=sp.mean(second_half + diff_mu), scale=5, size=len(second_half))), np.linspace(0, 1, len(second_half), endpoint=False), linewidth=3.0, color="orange")
    if mu_1 < mu_2:
        diff_mu = mu_2 - mu_1
        plt.plot(np.sort(stats.norm.rvs(loc=sp.mean(first_half + diff_mu), scale=5, size=len(first_half))), np.linspace(0, 1, len(first_half), endpoint=False), linewidth=3.0, color="g")
        plt.plot(np.sort(stats.norm.rvs(loc=second_half, scale=5, size=len(second_half))), np.linspace(0, 1, len(second_half), endpoint=False), linewidth=3.0, color="orange")
        
    plt.legend('top right')
    plt.legend(['First half', 'Second half'])
    plt.title('Comparing ' + str(field) + ' first half and fecond half ECDFs')
    plt.xticks([])
    plt.show()


# In[52]:


distribution('stator_yoke')


# In[53]:


sk_plot_norm('stator_yoke')


# In[54]:


cdf('stator_yoke')


# In[55]:


first_half, second_half = sk_test_2samp('stator_yoke')


# In[56]:


ks_plot_comp_cdf('stator_yoke', first_half, second_half)


# In[57]:


ks_plot_comp_ecdf('stator_yoke', first_half, second_half)


# In[58]:


distribution('stator_winding')


# In[59]:


sk_plot_norm('stator_winding')


# In[60]:


cdf('stator_winding')


# In[61]:


first_half, second_half = sk_test_2samp('stator_winding')


# In[62]:


ks_plot_comp_cdf('stator_winding', first_half, second_half)


# In[63]:


ks_plot_comp_ecdf('stator_winding', first_half, second_half)


# In[64]:


distribution('stator_tooth')


# In[65]:


sk_plot_norm('stator_tooth')


# In[66]:


cdf('stator_tooth')


# In[67]:


first_half, second_half = sk_test_2samp('stator_tooth')


# In[68]:


ks_plot_comp_cdf('stator_tooth', first_half, second_half)


# In[69]:


ks_plot_comp_ecdf('stator_tooth', first_half, second_half)


# In[70]:


distribution('pm')


# In[71]:


sk_plot_norm('pm')


# In[72]:


cdf('pm')


# In[73]:


first_half, second_half = sk_test_2samp('pm')


# In[74]:


ks_plot_comp_cdf('pm', first_half, second_half)


# In[75]:


ks_plot_comp_ecdf('pm', first_half, second_half)


# Visto que o valor p assume sempre o valor 0, i.e., valores tão baixos que arrendodados dão 0, é então rejeitada a hipotése nula. 
# As distribuições de dados não seguem qualquer distribuição conhecida.

# # Evaluate Algorithms

# ## Split-out validation dataset
# 
# Nesta fase é feita a separação das features e do target. Também é feita a divisão dos de treino (80%) e de teste (20%).

# Primeiro separa-se as features do target e normaliza-se os dados das features.

# In[323]:


electric_motor_temp = electric_motor_temp.loc[:,['stator_yoke', 'stator_winding', 'stator_tooth', 'pm']]
features = electric_motor_temp.drop('pm', 1)
target = electric_motor_temp['pm']

features = (features - np.min(features))/(np.max(features)-np.min(features))


# De seguida divide-se os dados em dados de treino e de teste como já referido.

# In[324]:


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# ## Spot Check Algorithms
# 
# Nesta fase inicial iremos utilizar os seguintes modelos:
# 
# - Linear Regression
# - Lasso Regression
# - Ridge Regression
# - Elastic Net Regression
# - Decision Tree Regressor
# - K-Nearest Neighbors Regressor
# - Neural Network Regressor

# Iremos agora definir esses modelos numa lista.

# In[78]:


models = []
models.append(('Linear Regression', LinearRegression(fit_intercept=False, n_jobs = -1)))
models.append(('Lasso', Lasso(alpha=0.1)))
models.append(('Ridge', Ridge(alpha=0.1)))
models.append(('Elastic Net', ElasticNet(alpha=0.1)))
models.append(('Decision Tree', DecisionTreeRegressor(random_state = 42)))
models.append(('KNN', KNeighborsRegressor(n_neighbors=2, n_jobs = -1)))
models.append(('Neural Network', MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='adam', verbose=0)))


# ## Test options and evaluation metric
# 
# Função com as métricas para os modelos definidos.
# É aplicado o cross-validation.
# As métricas utilizadas são:
# 
# - Mean Absolute Error
# - Mean Squared Error
# - R-Squared
# 
# Para comparação dos algoritmos iremos utilizar o R-Squared.

# In[79]:


def metrics(models):
    results = []
    names = []
    for name, model in models:
        model.fit(x_train, y_train)
        predict_train = model.predict(x_train)
        predict_test = model.predict(x_test)
        results.append(r2_score(y_test, predict_test))
        names.append(name)
        
        print(name + ':')
        print('\tTrain:')
        print('\t\tMean Absolute Error: {}'.format(mean_absolute_error(y_train, predict_train)))
        print('\t\tMean Squared Error: {}'.format(mean_squared_error(y_train, predict_train)))
        print('\t\tR-Squared: {}'.format(r2_score(y_train, predict_train)))
        print('\tTest:')
        print('\t\tMean Absolute Error: {}'.format(mean_absolute_error(y_test, predict_test)))
        print('\t\tMean Squared Error: {}'.format(mean_squared_error(y_test, predict_test)))
        print('\t\tR-Squared: {}\n'.format(r2_score(y_test, predict_test)))
        
    return names, results


# In[80]:


models_names, models_results = metrics(models)


# ### Cross-validation
# 
# Vamos utilizar o cross-validation para avaliar a capacidade de generalização dos modelos.
# A estratégia aplicada para validar o modelo foi o cross-validation Shuffle Split, que é um iterador recomendado para conjuntos de dados com um grande desequilíbrio na sua distribuição.
# Dividi o conjunto em 10 grupos (splits), por 10 vezes, dos quais 8 foram usados para treino e 2 foram usados para teste.
# O cross-validation é computacionalmente mais pesado, mas vale a pena o esforço.

# In[325]:


shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)


# Função com as métricas utilizando cross-validation.

# In[82]:


def metrics_cv(models, with_pca):
    results = []
    names = []
    for name, model in models:
        if not with_pca:
            model = make_pipeline(preprocessing.StandardScaler(), model)
        cv_mae = cross_val_score(model, features, target, cv=shufflesplit, scoring='neg_mean_absolute_error')
        cv_mse = cross_val_score(model, features, target, cv=shufflesplit, scoring='neg_mean_squared_error')
        cv_r2 = cross_val_score(model, features, target, cv=shufflesplit, scoring='r2')
        results.append(cv_r2.mean())
        names.append(name)
        
        print(name + ':')
        print('\tMean Absolute Error: {} ({})'.format(cv_mae.mean(), cv_mae.std()))
        print('\tMean Squared Error: {} ({})'.format(cv_mse.mean(), cv_mse.std()))
        print('\tR-Squared: {} ({})\n'.format(cv_r2.mean(), cv_r2.std()))
    
    return names, results


# In[83]:


models_names_cv, models_results_cv = metrics_cv(models, False)


# ## Compare Algorithms
# 
# Função que constroi um gráfico com a comparação dos resultados dos algoritmos.

# In[84]:


def compare_algorithms(names, results, title):
    x = names
    y = np.array(results)
    length = len(names)
    models = pd.DataFrame({'model': names, 'results': results})
    models.set_index('model', inplace=True)
    models.sort_values('results', ascending = False).plot(y = 'results', kind = 'bar', legend=False, title = title).grid(axis='y')


# In[85]:


compare_algorithms(models_names_cv, models_results_cv, 'Comparaçao dos Algoritmos com cross-validation')


# Nesta fase os algoritmos KNN e Decision Tree destacam-se dos demais.

# # Dimensionality reduction - PCA

# Para executar a redução de dimensionalidade é necessário fazer primeiro a normalização escalar padrão.

# In[180]:


features = electric_motor_temp.drop('pm', 1).values
target = electric_motor_temp['pm'].values
sc = StandardScaler()
features = sc.fit_transform(features)


# Precisamos de saber qual o número de componentes que iremos utilizar.

# In[181]:


pca_train = PCA().fit(features)
plt.plot(np.cumsum(pca_train.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# Função que devolve o número de componentes ideal.

# In[182]:


def select_n_components(var_ratio, goal_var: float) -> int:
    total_variance = 0.0
    n_components = 0
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break

    return n_components


# In[183]:


pca_var_ratio = pca_train.explained_variance_ratio_
n_components = select_n_components(pca_var_ratio, 0.99)
print('O número de componentes é ' + str(n_components))
pca = PCA(n_components=n_components)
features = pca.fit_transform(features)
features = pd.DataFrame(data = features)


# In[184]:


perc = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
fields = ['PC' + str(x) for x in range(1, len(perc) + 1)]
plt.bar(x=range(1, len(perc)+1), height=perc, tick_label=fields)
plt.ylabel('Percentange of explained variance')
plt.xlabel('Principal Component')
plt.title('PCA')
plt.show()


# Para uma validação completa, é averiguado os resultados dos algoritmos com cross-validation e PCA.

# In[91]:


pca_models_names, pca_models_results = metrics_cv(models, True)


# In[92]:


compare_algorithms(pca_models_names, pca_models_results, 'Comparaçao dos Algoritmos com PCA')


# Aplicando a redução da dimensionalidade com o PCA, continua-se a ter os algoritmos KNN e Decision Tree como os que apresentam melhores resultados.

# # Improve Accuracy
# 
# Nesta fase ir-se-á tentar melhorar os resultados dos 2 melhores algoritmos anteriores.

# ## Algorithm Tuning
# 
# Para tentar melhorar os resultados, é feita uma "busca" pelos melhores parâmetros de cada modelo.
# Após encontrar os melhores parâmetros para cada modelo, são criados os modelos com esses parâmetros de forma automática.

# ### K-Nearest Neighbors Regressor
# 
# Para o KNN testa-se qual o número de k que apresenta melhores resultados.

# In[93]:


def r2_k_knn():
    r2_list = []
    for k in range(15):
        k = k+1
        model = KNeighborsRegressor(n_neighbors = k)
        shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        cv_r2 = cross_val_score(model, features, target, cv=shufflesplit, scoring='r2')
        r2_list.append(cv_r2.mean())
        print('K = {}, R-Squared: {}'.format(k, cv_r2.mean()))
        
    return r2_list


# In[94]:


r2_list = r2_k_knn()


# In[95]:


def plot_n_neighbors(r2_list):
    curve = pd.DataFrame(r2_list)
    curve.plot()


# In[96]:


plot_n_neighbors(r2_list)


# In[97]:


def best_k():
    n_neighbors = list(range(1,10))
    weights = ['uniform', 'distance']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size = list(range(1,30))
    p = [1, 2, 3, 4, 5]
    random_grid = {'n_neighbors': n_neighbors, 'weights': weights, 'algorithm': algorithm, 'leaf_size': leaf_size, 'p': p}
    rf_random = RandomizedSearchCV(estimator = KNeighborsRegressor(), param_distributions = random_grid, n_iter = 10, cv = shufflesplit, verbose=2, random_state = 42, n_jobs = -1)
    rf_random.fit(features,target)
    print(rf_random.best_params_)
    return rf_random.best_params_


# In[98]:


knn_best_params = best_k()


# ### Decision Tree
# 
# Para o Decision Tree testa-se vários parâmetros que estão descritos abaixo para averiguar quais apresentam melhores resultados.

# In[99]:


def decision_tree_tuning():
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [1, 2, 4, 8]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [2,5,10,20,40,80,100,200,400,800,1600,32000,64000]
    random_grid = {'max_features': max_features, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split}
    rf_random = RandomizedSearchCV(estimator = DecisionTreeRegressor(), param_distributions = random_grid, n_iter = 10, cv = shufflesplit, verbose=2, random_state = 42, n_jobs = -1)
    rf_random.fit(features,target)
    print(rf_random.best_params_)
    return rf_random.best_params_


# In[100]:


decision_tree_best_params = decision_tree_tuning()


# Tendo já os melhores parâmetros para cada modelo, é criada uma lista com esses algortimos e esses parâmetros.

# In[101]:


tuning_models = []
tuning_models.append(('KNN', KNeighborsRegressor(weights = knn_best_params.get('weights'), p = knn_best_params.get('p'), n_neighbors = knn_best_params.get('n_neighbors'), leaf_size = knn_best_params.get('leaf_size'), algorithm = knn_best_params.get('algorithm'), n_jobs = -1)))
tuning_models.append(('Decision Tree', DecisionTreeRegressor(min_samples_split = decision_tree_best_params.get('min_samples_split'), min_samples_leaf = decision_tree_best_params.get('min_samples_leaf'), max_features = decision_tree_best_params.get('max_features'), max_depth = decision_tree_best_params.get('max_depth'), random_state = 42)))


# In[102]:


tuning_models_names, tuning_models_results = metrics_cv(tuning_models, True)


# In[103]:


compare_algorithms(tuning_models_names, tuning_models_results, 'Comparaçao dos 2 melhores Algoritmos com Tuning')


# Claramente existiu uma melhoria nos resultados, sendo essas melhorias as seguintes:
# 
# - KNN: 0.9251857823656312 -> 0.9339753628739441
# - Decision Tree: 0.8837846649885744 -> 0.910290796166594

# ## Ensembles
# 
# Para além dos algoritmos anteriores, ir-se-á testar alguns algoritmos e técnicas Ensembles.
# 
# Bagging algorithms:
# 
# - Random Forest Regressor
# - Extra Trees Regressor
# 
# Boosting algorithms:
# 
# - Adaboost Regressor
# - Gradient Boosting Regressor
# - XGBoost Regressor

# É criada uma lista com esses modelos.

# In[104]:


ensembles_models = []
# Bagging algorithms
ensembles_models.append(('Random Forest', RandomForestRegressor(n_estimators = 10, random_state = 42, n_jobs=-1)))
ensembles_models.append(('Extra Trees', ExtraTreesRegressor(n_estimators=10, n_jobs=-1, random_state=42)))
# Boosting algorithms
ensembles_models.append(('AdaBoost', AdaBoostRegressor(n_estimators=10, random_state=42)))
ensembles_models.append(('Gradient Boosting', GradientBoostingRegressor(random_state=42)))
ensembles_models.append(('XGBoost', xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)))


# In[105]:


ensembles_models_names, ensembles_models_results = metrics_cv(ensembles_models, True)


# In[106]:


compare_algorithms(ensembles_models_names, ensembles_models_results, 'Comparaçao dos Algoritmos Ensembles')


# Claramente que os Bagging algorithms - Extra Trees e Random Forest foram os que apresentaram melhores resultados.

# ### Ensembles Tuning
# 
# Para tentar melhorar os resultados dos algoritmos Ensembles, é feita uma "busca" pelos melhores parâmetros de cada algortimo.

# #### Random Forest Regressor

# In[107]:


def random_forest_tuning():
    n_estimators = [200, 300, 500]
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [1, 2, 4, 8]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [2,5,10,20,40,80,100,200,400,800,1600,32000,64000]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = random_grid, n_iter = 10, cv = shufflesplit, verbose=2, random_state = 42, n_jobs = -1)
    rf_random.fit(features,target)
    print(rf_random.best_params_)
    return rf_random.best_params_


# In[108]:


random_forest_best_params = random_forest_tuning()


# #### Extra Trees Regressor

# In[109]:


def extra_trees_tuning():
    n_estimators = [200, 300, 500]
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [1, 2, 4, 8]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [2,5,10,20,40,80,100,200,400,800,1600,32000,64000]
    random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split}
    rf_random = RandomizedSearchCV(estimator = ExtraTreesRegressor(), param_distributions = random_grid, n_iter = 10, cv = shufflesplit, verbose=2, random_state = 42, n_jobs = -1)
    rf_random.fit(features,target)
    print(rf_random.best_params_)
    return rf_random.best_params_


# In[110]:


extra_trees_best_params = extra_trees_tuning()


# Tendo já os melhores parâmetros para cada modelo, é criada uma lista com esses algortimos e esses parâmetros.

# In[111]:


tuning_ensembles_models = []
tuning_ensembles_models.append(('Random Forest', RandomForestRegressor(n_estimators = random_forest_best_params.get('n_estimators'), min_samples_split = random_forest_best_params.get('min_samples_split'), min_samples_leaf = random_forest_best_params.get('min_samples_leaf'), max_features = random_forest_best_params.get('max_features'), max_depth = random_forest_best_params.get('max_depth'), bootstrap = random_forest_best_params.get('bootstrap'), random_state = 42, n_jobs=-1)))
tuning_ensembles_models.append(('Extra Trees', ExtraTreesRegressor(n_estimators = extra_trees_best_params.get('n_estimators'), min_samples_split = extra_trees_best_params.get('min_samples_split'), min_samples_leaf = extra_trees_best_params.get('min_samples_leaf'), max_features = extra_trees_best_params.get('max_features'), max_depth = extra_trees_best_params.get('max_depth'), n_jobs = -1, random_state = 42)))


# In[112]:


tuning_ensembles_models_names, tuning_ensembles_models_results = metrics_cv(tuning_ensembles_models, True)


# In[113]:


compare_algorithms(tuning_ensembles_models_names, tuning_ensembles_models_results, 'Comparaçao dos Algoritmos Ensembles com Tuning')


# Existiu uma melhoria nos resultados, sendo essas melhorias as seguintes:
# 
# - Random Forest: 0.9254653899128537 -> 0.9284612376463681 
# - Extra Trees: 0.9285981537035457 -> 0.9345996616082312  

# ### Voting Regressor
# 
# É utilizada a técnica de Ensemble - Voting Regressor com os 2 melhores modelos Ensembles com os parâmetros que apresentam os melhores resultados.

# In[114]:


def voting_regressor(models):
    ensemble = VotingRegressor(models)
    cv_mae = cross_val_score(ensemble, features, target, cv=shufflesplit, scoring='neg_mean_absolute_error')
    cv_mse = cross_val_score(ensemble, features, target, cv=shufflesplit, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(ensemble, features, target, cv=shufflesplit, scoring='r2')
    name = ['Voting Regressor']
    result = [cv_r2.mean()]
        
    print('Voting Regressor with Random Forest and Extra Trees:')
    print('\tMean Absolute Error: {} ({})'.format(cv_mae.mean(), cv_mae.std()))
    print('\tMean Squared Error: {} ({})'.format(cv_mse.mean(), cv_mse.std()))
    print('\tR-Squared: {} ({})\n'.format(cv_r2.mean(), cv_r2.std()))
    
    return name, result


# In[115]:


voting_regressor_models_name, voting_regressor_models_result = voting_regressor(tuning_ensembles_models)


# A técnica - Voting Regressor com os 2 melhores modelos Ensembles com os parâmetros que apresentam os melhores resultados, apresentou melhores resultados que o Random Forest mas apresentou piores do que o Extra Trees.

# ### Bagging Regressor
# 
# É utilizada a técnica de Ensemble - Bagging Regressor com os 2 melhores modelos iniciais com os parâmetros que apresentam os melhores resultados.

# In[116]:


def bagging_regressor(models):
    results = []
    names = []
    for name, model in models:
        model = BaggingRegressor(model)
        shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        cv_mae = cross_val_score(model, features, target, cv=shufflesplit, scoring='neg_mean_absolute_error')
        cv_mse = cross_val_score(model, features, target, cv=shufflesplit, scoring='neg_mean_squared_error')
        cv_r2 = cross_val_score(model, features, target, cv=shufflesplit, scoring='r2')
        results.append(cv_r2.mean())
        names.append('Bagging ' + str(name))
        
        print('Bagging ' + str(name) + ':')
        print('\tMean Absolute Error: {} ({})'.format(cv_mae.mean(), cv_mae.std()))
        print('\tMean Squared Error: {} ({})'.format(cv_mse.mean(), cv_mse.std()))
        print('\tR-Squared: {} ({})\n'.format(cv_r2.mean(), cv_r2.std()))
    
    return names, results


# In[117]:


bagging_models_names, bagging_models_results = bagging_regressor(tuning_models)


# In[118]:


compare_algorithms(bagging_models_names, bagging_models_results, 'Comparaçao dos Algoritmos com Bagging Regressor')


# Com a utilização desta técnica - Bagging meta-estimator permitiu melhorar os resultados.
# 
# - KNN: 0.9339753628739441 -> Bagging with KNN: 0.934259175790204
# - Decision Tree: 0.910290796166594 -> Bagging with Decision Tree: 0.9225060818924635

# ### Bagging Regressor Tuning
# 
# Mais uma vez, é utilizado a mesma técnica para tentar encontrar os melhores parâmetros que resultam nos melhores resultados possíveis, mas desta vez para os algoritmos 2 melhores algoritmos com Bagging Regressor.

# #### Bagging KNN Regressor

# In[119]:


def bagging_regressor_knn_tuning():
    model = KNeighborsRegressor(weights = knn_best_params.get('weights'), p = knn_best_params.get('p'), n_neighbors = knn_best_params.get('n_neighbors'), leaf_size = knn_best_params.get('leaf_size'), algorithm = knn_best_params.get('algorithm'), n_jobs = -1)
    n_estimators = [200, 300, 500]
    max_samples = [0.25, 0.5, 0.75, 1.0]
    max_features = [0.25, 0.5, 0.75, 1.0]
    bootstrap = [True, False]
    bootstrap_features = [True, False]
    warm_start = [True, False]
    random_grid = {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features, 'bootstrap': bootstrap, 'bootstrap_features': bootstrap_features, 'warm_start': warm_start}
    rf_random = RandomizedSearchCV(estimator = BaggingRegressor(base_estimator = model), param_distributions = random_grid, n_iter = 10, cv = shufflesplit, verbose=2, random_state = 42, n_jobs = -1)
    rf_random.fit(features,target)
    print(rf_random.best_params_)
    
    
    return rf_random.best_params_


# In[120]:


bagging_knn_best_params = bagging_regressor_knn_tuning()


# #### Bagging Decision Tree Regressor

# In[121]:


def bagging_regressor_decision_tree_tuning():
    model = DecisionTreeRegressor(min_samples_split = decision_tree_best_params.get('min_samples_split'), min_samples_leaf = decision_tree_best_params.get('min_samples_leaf'), max_features = decision_tree_best_params.get('max_features'), max_depth = decision_tree_best_params.get('max_depth'), random_state = 42)
    n_estimators = [200, 300, 500]
    max_samples = [0.25, 0.5, 0.75, 1.0]
    max_features = [0.25, 0.5, 0.75, 1.0]
    bootstrap = [True, False]
    bootstrap_features = [True, False]
    warm_start = [True, False]
    random_grid = {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features, 'bootstrap': bootstrap, 'bootstrap_features': bootstrap_features, 'warm_start': warm_start}
    rf_random = RandomizedSearchCV(estimator = BaggingRegressor(base_estimator = model), param_distributions = random_grid, n_iter = 10, cv = shufflesplit, verbose=2, random_state = 42, n_jobs = -1)
    rf_random.fit(features,target)
    print(rf_random.best_params_)
    return rf_random.best_params_


# In[122]:


bagging_decision_tree_best_params = bagging_regressor_decision_tree_tuning()


# É criada uma lista com os algoritmos e os melhores parâmetros.

# In[123]:


knn = KNeighborsRegressor(weights = knn_best_params.get('weights'), p = knn_best_params.get('p'), n_neighbors = knn_best_params.get('n_neighbors'), leaf_size = knn_best_params.get('leaf_size'), algorithm = knn_best_params.get('algorithm'), n_jobs = -1)
decision_tree = DecisionTreeRegressor(min_samples_split = decision_tree_best_params.get('min_samples_split'), min_samples_leaf = decision_tree_best_params.get('min_samples_leaf'), max_features = decision_tree_best_params.get('max_features'), max_depth = decision_tree_best_params.get('max_depth'), random_state = 42)
tuning_bagging_regressor_models = []
tuning_bagging_regressor_models.append(('Bagging KNN', BaggingRegressor(base_estimator = knn, n_estimators = bagging_knn_best_params.get('n_estimators'), max_samples = bagging_knn_best_params.get('max_samples'), max_features = bagging_knn_best_params.get('max_features'), bootstrap = bagging_knn_best_params.get('bootstrap'), bootstrap_features = bagging_knn_best_params.get('bootstrap_features'), warm_start = bagging_knn_best_params.get('warm_start'), random_state = 42)))
tuning_bagging_regressor_models.append(('Bagging Decision Tree', BaggingRegressor(base_estimator = decision_tree, n_estimators = bagging_decision_tree_best_params.get('n_estimators'), max_samples = bagging_decision_tree_best_params.get('max_samples'), max_features = bagging_decision_tree_best_params.get('max_features'), bootstrap = bagging_decision_tree_best_params.get('bootstrap'), bootstrap_features = bagging_decision_tree_best_params.get('bootstrap_features'), warm_start = bagging_decision_tree_best_params.get('warm_start'), random_state = 42)))


# In[124]:


tuning_bagging_models_names, tuning_bagging_models_results = metrics_cv(tuning_bagging_regressor_models, True)


# In[125]:


compare_algorithms(tuning_bagging_models_names, tuning_bagging_models_results, 'Comparaçao dos Algoritmos com Bagging e com Tuning')


# Utilizando os melhores parâmetros, os resultados pouco se alteraram:
# 
# - Bagging with KNN: 0.934259175790204 -> 0.9342204342605417
# - Bagging with Decision Tree: 0.9225060818924635 -> 0.9250700833641424

# É criada uma lista com os melhores algortimos Ensembles com os parâmetros que apresentam melhores resultados.
# 
# - Bagging algorithms
# - Voting Regressor com os Bagging algorithms
# - Bagging KNN
# - Bagging Decision Tree

# In[126]:


best_models_results = tuning_ensembles_models_results + voting_regressor_models_result + tuning_bagging_models_results
best_models_names = tuning_ensembles_models_names + voting_regressor_models_name + tuning_bagging_models_names


# In[127]:


compare_algorithms(best_models_names, best_models_results, 'Comparaçao dos melhores Algoritmos')


# Todos os modelos apresentam resultados muito bons, mas escolhendo apenas os 2 melhores, são escolhidos o Extra Trees e o Bagging KNN.

# # Finalize Model
# 
# Nesta fase final é criado o modelo final com o Voting Regressor, utilizando o Extra Trees e o Bagging KNN.

# É criada uma lista com o Voting Regressor, utilizando o Extra Trees e o Bagging KNN.

# In[426]:


best_models = []
best_models.append(('Extra Trees', ExtraTreesRegressor(n_estimators = extra_trees_best_params.get('n_estimators'), min_samples_split = extra_trees_best_params.get('min_samples_split'), min_samples_leaf = extra_trees_best_params.get('min_samples_leaf'), max_features = extra_trees_best_params.get('max_features'), max_depth = extra_trees_best_params.get('max_depth'), n_jobs = -1, random_state = 42)))
best_models.append(('Bagging KNN', BaggingRegressor(base_estimator = knn, n_estimators = bagging_knn_best_params.get('n_estimators'), max_samples = bagging_knn_best_params.get('max_samples'), max_features = bagging_knn_best_params.get('max_features'), bootstrap = bagging_knn_best_params.get('bootstrap'), bootstrap_features = bagging_knn_best_params.get('bootstrap_features'), warm_start = bagging_knn_best_params.get('warm_start'), random_state = 42, n_jobs=-1)))
final_model = VotingRegressor(best_models)


# Existe a necessidade de nomear as features e o target outra vez e criar a divisão de treino e de teste, para não se utilizar o PCA, porque o objetivo é guardar o treino das 3 features e não apenas 2, isto porque o PCA utiliza apenas 2 componentes.

# In[427]:


features = electric_motor_temp.drop('pm', 1)
target = electric_motor_temp['pm']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# ## Predictions on validation dataset
# 
# Validação final ao modelo com as métricas e a validação de 20 registos aleatórios.

# In[430]:


def final_predictions(model):
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
        
    print('Test Final Model:')
    print('\t\tMean Absolute Error: {}'.format(mean_absolute_error(y_test, predict)))
    print('\t\tMean Squared Error: {}'.format(mean_squared_error(y_test, predict)))
    print('\t\tR-Squared: {}\n'.format(r2_score(y_test, predict)))
    
    print('Test 20 temperatures:\n')
    for i in range(20):
        i = random.choice(x_test.index.tolist())
        prediction = model.predict(x_test.loc[[i],:])
        prediction_rounded = round(prediction[0], 3)
        real_value_rounded = round(y_test[i], 3)
        print('\tReal Value: ' + str(real_value_rounded) + ', Predicted Value: ' + str(prediction_rounded) + '\n')

    return predict


# In[431]:


predict = final_predictions(final_model)


# In[409]:


def plot_predictions(predict):
    sns.kdeplot(predict, label = 'Predictions')
    sns.kdeplot(y_test, label = 'Real Values')
    plt.xlabel('PM')
    plt.ylabel('Density')
    plt.title('Test Predictions')


# In[410]:


plot_predictions(predict)


# Visualizando os resultados e o gráfico, de facto este modelo tem uma margem de erro pequena e as predições ajustam-se bem aos valores de teste.

# ## Create standalone model on entire training dataset and Save model for later use
# 
# Por fim é guardado o modelo treinado para ser utilizado quando se quiser. É treinado com todo o conjunto de dados.

# In[411]:


def save_model(model, file):
    model.fit(features, target)
    pickle.dump(model, open(file, 'wb'))
    print('Model saved!')


# In[412]:


save_model(final_model, 'final_model.pkl')


# # Load and use the final model
# 
# Agora é possível carregar o modelo e prever com outros dados, neste caso importados por um novo ficheiro.

# In[416]:


df_test = pd.read_csv('electric_car_temp_test.csv')
df_test = df_test.loc[:,['stator_yoke', 'stator_winding', 'stator_tooth', 'pm']]
features = df_test.drop('pm', 1)
target = df_test['pm']


# In[422]:


def predict(file):
    model = pickle.load(open(file, 'rb'))
    pred = model.predict(features)
    print('Final model:\n')
    for i in range(len(target)):
        prediction = model.predict(features.loc[[i],:])
        prediction_rounded = round(prediction[0], 3)
        real_value_rounded = round(target[i], 3)
        print('\tReal Value: ' + str(real_value_rounded) + ', Predicted Value: ' + str(prediction_rounded) + '\n')


# In[423]:


predict('final_model.pkl')

