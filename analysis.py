
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

# you need to run the following line in your console to install the package:
# pip install nelson-siegel-svensson
# more documentation see: https://nelson-siegel-svensson.readthedocs.io/en/latest/
from nelson_siegel_svensson import NelsonSiegelCurve
#from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


params = {'legend.fontsize': 'medium',
          'figure.figsize': (10, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

#%%
# DATA IMPORTING AND CLEANING

path = r'D:\Studies\UT\MFI\STA2540 Ins Risk\Lin\project\data'

df_full = pd.read_csv(path+"\DailyTreasuryYieldCurveRateData.csv")
# 2 months rates are only available from 2018 and onward
df = df_full.drop(columns=['Date', '2 Mo'])
#df.isnull().sum()

# filling a missing value with previous ones   
#df = df.fillna(method ='pad') 

# interpolate the missing values using Linear method
# ignore the index and treat the values as equally spaced.
df = df.interpolate(method ='linear', limit_direction ='forward') 
#df.isnull().any()

#%%
# DATA EXPLORATION

df_full[['1 Mo','2 Mo','3 Mo']].plot(figsize=(10,6))

df.plot(figsize=(12,8), legend='reverse')

df.plot.box(figsize=(10,6))

#%%
fig = plt.figure(figsize=(10,6))
ax = fig.gca(projection='3d')

t = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
Y = np.linspace(2007, 2020, len(df))
X, Y = np.meshgrid(t, Y)
Z = np.array(df)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.title('Treasury Yield Surface')
ax.set_yticks([2007, 2011, 2015, 2020])
ax.set_xlabel('Tenor')
ax.set_ylabel('Year')
ax.set_zlabel('Yield (%)')
#fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(30, -140)
plt.show()

#%%
# NS

#y = NelsonSiegelSvenssonCurve(0.028, -0.03, -0.04, -0.015, 1.1, 4.0)
#t = np.linspace(0, 30, 100)
#plt.plot(t, y(t))


#t = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
#y = np.array([0.01, 0.011, 0.013, 0.016, 0.019, 0.021, 0.026, 0.03, 0.035, 0.037, 0.038, 0.04])
#
#curve, status = calibrate_ns_ols(t, y, tau0=1.0)  # starting value of 1.0 for the optimization of tau
#assert status.success
#print(curve)


# =============================================================================
# t = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
# 
# # Z is just our data in array form, that was created in graph section
# #curve, status = calibrate_ns_ols(t, Z, tau0=0.0229)
# 
# y = Z[-1]
# curve, status = calibrate_ns_ols(t, y)
# assert status.success
# print(curve)
# 
# x_t = np.linspace(0, 30, 100)
# plt.figure(figsize=(10,6))
# plt.plot(x_t, curve(x_t))
# plt.scatter(t, y)
# plt.show()
# =============================================================================


def ns_estimate (y, graph_curve=True, t=np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])):
    curve, status = calibrate_ns_ols(t, y)
    assert status.success
    
    if graph_curve:
        print(curve)
        x_t = np.linspace(0, t[-1], 100)
        plt.figure(figsize=(8,5))
        plt.plot(x_t, curve(x_t), label='NS curve')
        plt.scatter(t, y, label='True yield')
        plt.legend()
        plt.show()
    
    mse = mean_squared_error(y, curve(y))
    
    return mse
    

#%%
print('MSE is ', ns_estimate(Z[-1]))
# use accumulated MSE on test data sets to have a score maybe

#%%
# PCA

z_stand = StandardScaler().fit_transform(df)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(z_stand)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2', 'pc3'])


#principalDf.plot()
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)














