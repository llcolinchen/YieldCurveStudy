
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

# you need to run the following line in your console to install the package:
# pip install nelson-siegel-svensson
# more documentation see: 
# https://nelson-siegel-svensson.readthedocs.io/en/latest/
#from nelson_siegel_svensson import NelsonSiegelCurve
#from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
#from nelson_siegel_svensson.calibrate import calibrate_ns_ols

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

params = {'legend.fontsize': 'medium',
          'figure.figsize': (10, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

#%%
# DATA IMPORTING AND CLEANING

#path = r'D:\Studies\UT\MFI\STA2540 Ins Risk\Lin\project\data'
#df_full = pd.read_csv(path+"\DailyTreasuryYieldCurveRateData.csv")

dir_path = os.path.dirname(os.path.realpath(__file__))
#df_full = pd.read_csv(dir_path + '\data\DailyTreasuryYieldCurveRateData.csv')
df_full = pd.read_csv(dir_path + '\data\MonthlyTreasuryYieldCurveRateData.csv')


# 2 months rates are only available from 2018 and onward
df = df_full.drop(columns=['Date', '2 Mo'])
#df.isnull().sum()

# filling a missing value with previous ones   
#df = df.fillna(method ='pad') 

# interpolate the missing values using Linear method
# ignore the index and treat the values as equally spaced.
df = df.interpolate(method ='linear', limit_direction ='forward') 
df.isnull().any()

#df_input = df[['3 Mo', '5 Yr', '10 Yr']]
df_dy = df[['3 Mo', '5 Yr', '10 Yr']].diff().drop(0).reset_index(drop=True)
df_dy.columns = ['d_3 Mo', 'd_5 Yr', 'd_10 Yr']

#%%
# DATA EXPLORATION

df_full[['1 Mo','2 Mo','3 Mo']].plot(figsize=(10,6))
start = pd.to_datetime(df_full.Date.iloc[0]).year
end = pd.to_datetime(df_full.Date.iloc[-1]).year + 1


df.plot(figsize=(12,8), legend='reverse')

df.plot.box(figsize=(10,6))

#%%
#fig = plt.figure(figsize=(10,6))
#ax = fig.gca(projection='3d')
#
#t = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
#Y = np.linspace(2007, 2020, len(df))
#X, Y = np.meshgrid(t, Y)
#
#surf = ax.plot_surface(X, Y, df, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
#plt.title('Treasury Yield Surface')
#ax.set_yticks([2007, 2011, 2015, 2020])
#ax.set_xlabel('Tenor')
#ax.set_ylabel('Year')
#ax.set_zlabel('Yield (%)')
##fig.colorbar(surf, shrink=0.5, aspect=5)
#
#ax.view_init(30, -140)
#plt.show()

def plot_yield_surface(dat, start, end, title, yield_plot=True):
    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='3d')
    
    t = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    y = np.linspace(start, end, len(dat))
    X, Y = np.meshgrid(t, y)
    
    if yield_plot:
        surf = ax.plot_surface(X, Y, dat, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
    else:
        surf = ax.plot_surface(X, Y, dat,
                               linewidth=0, antialiased=False)
    
    plt.title(title)
#    ax.set_zlim(-0.01, 6.01)
    ax.set_yticks([start, int(np.quantile(y,1/3)), int(np.quantile(y,2/3)), end])
    ax.set_xlabel('Tenor')
    ax.set_ylabel('Year')
    ax.set_zlabel('Yield (%)')
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax.view_init(30, -140)
    plt.show()

#%%
plot_yield_surface(df, start, end, "Treasury Yield Surface")


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


# =============================================================================
# def ns_parameters (y, graph_curve=True, t=np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])):
#     curve, status = calibrate_ns_ols(t, y)
#     assert status.success
#     
#     if graph_curve:
#         print(curve)
#         x_t = np.linspace(0, 30, 100)
#         plt.figure(figsize=(8,5))
#         plt.plot(x_t, curve(x_t), label='NS curve')
#         plt.scatter(t, y, label='True yield')
#         plt.legend()
#         plt.show()
#     
#     rmse = np.sqrt(mean_squared_error(y, curve(y)))
#     
#     return rmse
# 
# print('RMSE is ', ns_parameters(df.iloc[-1]))
# =============================================================================

def ns_parameters (y, t=np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])):
    curve, status = calibrate_ns_ols(t, y)
    assert status.success
    
    [beta1, beta2, beta3, tau] = str(curve).split(',')
    beta1 = float(beta1.split('=')[1])
    beta2 = float(beta2.split('=')[1])
    beta3 = float(beta3.split('=')[1])
    tau = float(tau.split('=')[1][:-1])
    
    return beta1, beta2, beta3, tau


def get_I_matrix (tau, t=np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])):
#    assert len(y) == len(t), "input the correct schedule t for yields y"
    I = []
    
    for i in t:
        I1 = (1-np.exp(-i/tau)) / (i/tau)
        I2 = I1 - np.exp(i/tau)
        I.append([1, I1, I2])
    
    return np.array(I)


def ns_predict (y, dy, lmbda=0.0609):
#    b1, b2, b3, tau = ns_parameters(y)
    tau = 1/lmbda
    
    I_3 = get_I_matrix(tau, np.array([3/12, 5, 10]))
    I_11 = get_I_matrix(tau)
    
    dBeta = np.linalg.inv(I_3).dot(dy)
    dy_hat = I_11.dot(dBeta)
    
    y_pred = y + dy_hat
    
    return y_pred



def ns_main (dat, show_rmse=True, show_graph=True):
    
    rmse = np.empty(len(dat)-1)
    dat_pred = pd.DataFrame(columns=dat.columns)
    dat_pred = dat_pred.append(dat.iloc[0])
    
    for i in range(len(dat)-1):
#        print(i)
        y = dat.iloc[i]
        y_true = dat.iloc[i+1]
        dy = y_true - y
        dy = dy[['3 Mo', '5 Yr', '10 Yr']]
        y_pred = ns_predict(y, dy)
        
        dat_pred = dat_pred.append(y_pred)
        rmse[i] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if show_rmse:
        print("Average Root Mean Square Error of NS method is", round(np.mean(rmse),4))
        plt.figure(figsize=(10,7))
        plt.plot(rmse)
        plt.plot(np.repeat(np.mean(rmse), len(dat)-1))
        plt.title('Root Mean Square Error over Testing Periods')
        plt.ylabel('Errors')
        plt.xlabel('Periods')
        plt.show()
        
    if show_graph:
        plot_yield_surface(dat_pred, start, end, " NS Predicted Yield Surface")
        plot_yield_surface(dat_pred - dat, start, end, "Testing Errors for all periods", False)
    
    return dat_pred, rmse


#%%

ns_pred, ns_rmse = ns_main (df)



#y = df_input.iloc[-1]
#rmse = ns_parameters(y, t=np.array([3/12, 5, 10]))

# use accumulated RMSE on test data sets to have a score maybe

#%%
# PCA

def graph_var_exp (eig_vals):
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    
    
    plt.figure()
    plt.bar(np.arange(len(eig_vals)), var_exp, label='Individual')
    plt.plot(np.arange(len(eig_vals)), cum_var_exp, color='r', label='Cumulative')
    plt.xticks(np.arange(len(eig_vals)), ['PC%s' %i for i in range(1,len(eig_vals)+1)])
    plt.title('Explained variance by different principal components')
    plt.ylabel('Explained Variance (%)')
    plt.legend()
    plt.show()


def get_w (dat,graph_exp_var=False):
    z_stdized = StandardScaler().fit_transform(dat)
    cov_mat = np.cov(z_stdized.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()
    
    if graph_exp_var: graph_var_exp (eig_vals)

    matrix_w = np.hstack((eig_pairs[0][1].reshape(len(eig_vals),1), 
                          eig_pairs[1][1].reshape(len(eig_vals),1), 
                          eig_pairs[2][1].reshape(len(eig_vals),1)))
    
    return matrix_w



#dat = df.iloc[:11]
#dy = df_dy.iloc[10]
def pca_predict (dat, dy):
    matrix_w_full = get_w (dat)
    matrix_w = np.array([matrix_w_full[1],
                         matrix_w_full[6],
                         matrix_w_full[8]])
    d_pc = matrix_w.T.dot(dy)
    
#    dy_hat = np.linalg.inv(matrix_w_full).dot(d_pc)
#    dy_hat, _, _, _ = np.linalg.lstsq(matrix_w_full,d_pc)
    dy_hat = np.linalg.pinv(matrix_w_full.T).dot(d_pc)
#    dy_hat,_,_,_ = np.linalg.lstsq(matrix_w_full.T, d_pc)
    
    y_pred = dat.iloc[-1] + dy_hat
    
    return y_pred



def pca_main (dat, dat_dy, show_rmse=True, show_graph=True):
    
    skip = 10
    rmse = np.empty(len(dat)-1-skip)
    dat_pred = dat.copy().iloc[:skip] # PCA requires minimum data for covariance
    
    for i in range(skip, len(dat)-1):
#        print(i)
#        if i%100==0: print(i)
        dat_input = dat.copy().iloc[:i]
        y_true = dat.iloc[i+1]
        dy = dat_dy.iloc[i]
        y_pred = pca_predict(dat_input, dy)
        
        dat_pred = dat_pred.append(y_pred)
        rmse[i-skip] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if show_rmse:
        print("Average Root Mean Square Error of PCA method is", round(np.mean(rmse),4))
        plt.figure(figsize=(10,7))
        plt.plot(rmse)
        plt.plot(np.repeat(np.mean(rmse), len(dat)-1))
        plt.title('Root Mean Square Error over Testing Periods')
        plt.ylabel('Errors')
        plt.xlabel('Periods')
        plt.show()
    if show_graph:
        plot_yield_surface(dat_pred, start, end, " PCA Predicted Yield Surface")
        plot_yield_surface(dat_pred - df, start, end, "Testing Errors for all periods", False)
    
    return dat_pred, rmse

#%%
pca_pred, pca_rmse = pca_main (df, df_dy)

#%%


# =============================================================================
# z_stdized = StandardScaler().fit_transform(df)
# 
# cov_mat = np.cov(z_stdized.T)
# print('Covariance matrix \n%s' %cov_mat)
# 
# eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)
# 
# eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# eig_pairs.sort()
# eig_pairs.reverse()
# print('Eigenvalues in descending order:')
# for i in eig_pairs:
#     print(i[0])
# 
# graph_var_exp (eig_vals)
# 
# matrix_w = np.hstack((eig_pairs[0][1].reshape(len(eig_vals),1), 
#                       eig_pairs[1][1].reshape(len(eig_vals),1), 
#                       eig_pairs[2][1].reshape(len(eig_vals),1)))
# =============================================================================





#pca = PCA(n_components=3)
#principalComponents = pca.fit_transform(z_stand)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['pc1', 'pc2', 'pc3'])
#
##principalDf.plot()
#pca.explained_variance_ratio_
#sum(pca.explained_variance_ratio_)

# https://clinthoward.github.io/portfolio/2017/08/19/Rates-Simulations/


# https://plotly.com/python/v3/ipython-notebooks/principal-component-analysis/

# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60




#%%

# ANN


data = df.copy()
#data = data[:-1]

data["d_3 Mo"] = df_dy["d_3 Mo"]
data["d_5 Yr"] = df_dy["d_5 Yr"]
data["d_10 Yyr"] = df_dy["d_10 Yr"]


#data_std = StandardScaler().fit_transform(data) # Standardized data
#data_std = pd.DataFrame(data_std)

data_target = df.diff().drop(0).reset_index(drop=True)
#target_std = StandardScaler().fit_transform(data_target) # Standardized data
#target_std = pd.DataFrame(target_std)

#%%


class DNN_AE(nn.Module):
    def __init__(self, lr=0.1):
        super(DNN_AE, self).__init__()
        self.encoder = nn.Linear(14, 16)

        self.code = nn.Linear(16,3)

        self.decoder = nn.Linear(3,16)

        self.out = nn.Linear(16, 11)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
#        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        enc = F.tanh(self.encoder(input))

        code = self.code(enc)

        dec = F.tanh(self.decoder(code))

#        out = T.sigmoid(self.out(dec))
        out = F.tanh(self.out(dec))
#        out = self.out(dec)

        return out


#%%

class Learner(object):
    def __init__(self, DNN_AE, input, target, batch_size=1000, epochs=1000):
        self.DNN_AE = DNN_AE
        self.input = Variable(input)
        self.input_dim = input.shape
        self.target = Variable(target)
        self.target_dim = target.shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.current_batch = self.input[:self.batch_size,:]
        self.current_target = self.target[:self.batch_size,:]
        self.batch_num = 0
        self.batch_num_total = self.input_dim[0] // self.batch_size + (self.input_dim[0] % self.batch_size != 0)

    def reset(self):
        self.batch_num = 0
        self.current_batch = self.input[:self.batch_size,:]
        self.current_target = self.target[:self.batch_size,:]

    def next_batch(self):
        self.batch_num += 1
        self.current_batch = self.input[self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1),:]
        self.current_target = self.target[self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1),:]
        if self.batch_num == self.batch_num_total :
            self.reset()

    def learn(self):
        loss_hist = []
        
        for e in range(self.epochs) :
            epoch_loss = 0
            for b in range(self.batch_num_total):
                self.DNN_AE.optimizer.zero_grad()
                predictions = self.DNN_AE.forward(self.current_batch)
                loss = self.DNN_AE.loss(predictions, self.current_target)
#                print('epoch', str(e+1), '[ batch', str(b+1),'] - loss : ', str(loss.item()))
                loss.backward()
                self.next_batch()
                self.DNN_AE.optimizer.step()
            
                epoch_loss += loss.item()
            
            loss_hist.append(epoch_loss/self.batch_num_total)
            
            if e%100==0:
                print('epoch', str(e+1), ' - loss : ', str(loss_hist[-1]))
        
        plt.figure(figsize=(10,5))
        plt.plot(loss_hist)
        plt.title("Loss Histrogram")
        plt.show()   
        return self.DNN_AE


# =============================================================================
# #%%
# class Data:
#     def __init__(self, dat):
#         self.dir = dat
# 
#     def import_data(self):
#         data = pd.read_csv(self.dir)
#         data = self.normalize(data)
# 
#         target = data.iloc[:, 1:]
#         input = data.iloc[:, 1:]
#         target = T.Tensor(target.values)
#         input = T.Tensor(input.values)
#         return target, input
# 
#     def normalize(self, data):
#         return data/255.00
# 
# 
# =============================================================================
##%%
#class Visualizer :
#    def __init__(self, target, model):
#        self.target = target
#        self.model = model
#        self.samples = self.random_select()
#
#    def random_select(self):
#        rndm = np.random.randint(0,self.target.shape[0]-1,10)
#        return rndm
#
#    def viz(self):
#        for _, i in enumerate(self.samples):
#            output = self.model.forward(self.target[i]).view(28,28)
#            cat_img = T.cat((self.target[i].view(28,28),output), 1)
#            plt.imshow(cat_img.cpu().detach().numpy())
#            plt.title('DNN AutoEncoder target vs output')
#            plt.savefig('./output_images/sample'+str(_+1))
#            plt.close()



#%%


LEARNING_RATE = 0.01
#EPOCHS = 1000
EPOCHS = 3000
#BATCH_SIZE = 500
BATCH_SIZE = 100

#    target = target_std
#    input = data_std.iloc[:-1, :]
target = data_target
input = data.iloc[:-1, :]
target = T.Tensor(target.values)
input = T.Tensor(input.values)

net = DNN_AE(LEARNING_RATE)
target = target.to(net.device)
input = input.to(net.device)
print(target.shape)
print(input.shape)

learner = Learner(net, input, target, batch_size=BATCH_SIZE, epochs=EPOCHS)
model = learner.learn()

output = net.forward(input)
dy_pred = output.detach().numpy()
y_pred = data.iloc[:-1, :11].copy().add(dy_pred)
#y_pred = data_std.iloc[:-1, :11].copy().add(dy_pred)

ann_rmse = np.empty(len(df)-1)
for i in range(len(df)-1):
    y_true_ = df.iloc[i+1]
#    y_true_ = target_std.iloc[i]
    y_pred_ = y_pred.iloc[i]
    ann_rmse[i] = np.sqrt(mean_squared_error(y_true_, y_pred_))

print("Average Root Mean Square Error of ANN method is", round(np.mean(ann_rmse),4))
plt.figure(figsize=(10,7))
plt.plot(ann_rmse)
plt.plot(np.repeat(np.mean(ann_rmse), len(ann_rmse)))
plt.title('Root Mean Square Error over Testing Periods')
plt.ylabel('Errors')
plt.xlabel('Periods')
plt.show()

plot_yield_surface(y_pred, start, end, " ANN Predicted Yield Surface")
plot_yield_surface(y_pred - df.iloc[1:], start, end, "Testing Errors for all periods", False)

#viz = Visualizer(target, model)
#viz.viz()






# what learning rate to use
# fix batch things, that cuased bad loss histogram
# when should normalization happen? before we take diff or after




# https://github.com/imhgchoi/pytorch-implementations/blob/master/AutoEncoders/DNN_AE/main.py


#%%

# Prediction on 2020 Q1 senarios
senarios = pd.DataFrame({'3 Mo':[1.6, 0.1], '5 Yr': [1.7, 0.5], '10 Yr':[1.8, 0.7]})
senarios.index = ['Baseline', 'SA']

yeilds_2019 = df.iloc[-1][['3 Mo', '5 Yr', '10 Yr']]

senarios = senarios - yeilds_2019
senarios.columns = ['d_3 Mo', 'd_5 Yr', 'd_10 Yr']

#pred_bl = pd.DataFrame(columns=df.columns)
#pred_sa = pd.DataFrame(columns=df.columns)


ns_bl = ns_predict (df.iloc[-1], senarios.loc['Baseline'])
ns_sa = ns_predict (df.iloc[-1], senarios.loc['SA'])
#pred_bl = pred_bl.append(ns_bl, ignore_index=True)
#pred_sa = pred_sa.append(ns_sa, ignore_index=True)

pca_bl = pca_predict (df, senarios.loc['Baseline'])
pca_sa = pca_predict (df, senarios.loc['SA'])
#pred_bl = pred_bl.append(pca_bl, ignore_index=True)
#pred_sa = pred_sa.append(pca_sa, ignore_index=True)

input1 = df.iloc[-1].copy().append(senarios.loc['Baseline'])
input1 = T.Tensor(input1.values)
output1 = net.forward(input1)
dy_pred1 = output1.detach().numpy()
ann_bl = df.iloc[-1].copy().add(dy_pred1)
#pred_bl = pred_bl.append(ann_bl, ignore_index=True)

input2 = df.iloc[-1].copy().append(senarios.loc['SA'])
input2 = T.Tensor(input2.values)
output2 = net.forward(input2)
dy_pred2 = output2.detach().numpy()
ann_sa = df.iloc[-1].copy().add(dy_pred2)
#pred_sa = pred_sa.append(ann_sa, ignore_index=True)

#pred_bl.index = ['NS', 'PCA', 'ANN']
#pred_sa.index = ['NS', 'PCA', 'ANN']


plt.figure(figsize=(10,6))
plt.plot(ns_bl, label='NS')
plt.plot(pca_bl, label='PCA')
plt.plot(ann_bl, label='ANN')
plt.title('Predicted Yield Curve of 2020 Baseline Senario')
plt.ylabel('Yields (%)')
plt.legend()
plt.show()


plt.figure(figsize=(10,6))
plt.plot(ns_sa, label='NS')
plt.plot(pca_sa, label='PCA')
plt.plot(ann_sa, label='ANN')
plt.title('Predicted Yield Curve of 2020 Severely Adverse Senario')
plt.ylabel('Yields (%)')
plt.legend()
plt.show()


