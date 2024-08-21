# Python code for ICC conference paper entitled "SLIPT in Joint Dimming Multi-LED OWC Systems with Rate Splitting Multiple Access"

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import random
import math
from math import trunc
from model import PolicyNet, ValueNet
from agent import PPO
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

N_t = 6    #number of LEDs
num_on = 3   #number of On LEDs
K = 2      #number of users
si_c = 60
n_R = 1.5
fi_half = 60 #degree
A_VLC = 1e-4 #meter^2
I_h = 1e-2  #10mA
I_L = 0  # 0 A
I_zero = 5e-3  #5mA
I_dark_saturation = 1e-8 #10e-9A
V_t = 25e-3 #Volt
P_cir = 1e-3  #Watt
zeta = 1.2
phi_conversion_factor = 1
m = (-1*np.log(2))/(np.log(np.cos(np.pi/3)))
sig2_k = 1e-17
tau = 1
QoS = [2,2]
P_max = 1 # Watt
Har_Pow_PD = [1e-8, 1e-8] # Watt


LEDs_loc = np.array([[2,2,3],
                     [4,2,3],
                     [6,2,3],
                     [2,6,3],
                     [4,6,3],
                     [6,6,3]], dtype = float)
#Learning parameters----------------------------------------------
batch_size = 32
alpha=0.0000001
beta=0.0000001
gamma=0.99
tau=0.01
max_size=1000000 
fc1_dims=400
fc2_dims=300
# actor and critic hidden layers
C_fc1_dims = 1024
C_fc2_dims = 512
C_fc3_dims = 256

A_fc1_dims = 1024
A_fc2_dims = 512
memory_size = 1000000
#-------------------------------------------------------------
# users_loc = np.array([[4,1,1],
#                      [4,7,1]], dtype = float)
Action_size_PPO = K*N_t + N_t + K # w_P, w_C, R_star
state_size = (4*K)+1
#===================================================================
#===================================================================
def Loc_user():
    loc = np.zeros([K,3])
    for i in range(K):
        loc[i,0] = np.random.uniform(0, 8)
        loc[i,1] = np.random.uniform(0, 8)
        loc[i,2] = np.random.uniform(0, 3)
    return loc    
users_loc = Loc_user()        
#===================================================================
#===================================================================            
class si_fi:
    def __init__(self, users_location, LEDs_location, n_users, n_LEDs):
        self.n_users = n_users
        self.n_LEDs = n_LEDs
        self.users_location = users_location
        self.LEDs_location = LEDs_location
        
    def si(self):
        self.si_vec = np.zeros ([self.n_users,self.n_LEDs])
        self.dis_user_led = np.zeros ([self.n_users,self.n_LEDs])
        for i in range(self.n_users):
            for j in range(self.n_LEDs):
                self.dis_user_led[i,j] = np.sqrt( np.sum(np.abs(self.users_location[i] - self.LEDs_location[j] )**2))
                self.si_vec[i,j] = np.arccos( (self.LEDs_location[j,2] - self.users_location[i,2]) / self.dis_user_led[i,j]  )
        return   self.dis_user_led,  self.si_vec
#===================================================================
#===================================================================
zaviaha = si_fi(users_loc, LEDs_loc, K, N_t) 
distance_led_pd, Si = zaviaha.si()        
#===================================================================
#===================================================================               
class var:
    def __init__(self, K, N_t):
         self.N_t = N_t
         self.K = K

    def gen_s_c(self):
        s_c = (1 / np.sqrt(2)) * (np.random.randn())
        return s_c

    def gen_s_p(self):
        s_p = (1 / np.sqrt(2)) * (np.random.randn(self.K))
        s_p = s_p.reshape(self.K,1)
        return s_p
    
    def gen_w_c(self):
        W_c = np.zeros([self.N_t, 1])
        #W_i = np.eye(self.n_user_ref, dtype=complex)
        W_c = np.random.randn(self.N_t, 1) 
        return W_c    
    
    def gen_w_p(self):
        W_p = np.zeros([self.N_t, self.K])
        #W_i = np.eye(self.n_user_ref, dtype=complex)
        W_p = np.random.randn(self.N_t, self.K) 
        return W_p   
    
    def gen_F(self):
        F = np.random.randn(self.N_t, self.N_t)
        f = np.random.randint(2, size=self.N_t)
        for i in range(num_on):
            f[i] = 1
        for i in range(num_on,self.N_t):
            f[i] = 0 
        a = np.count_nonzero(f)
        if a == 0:
            f = np.random.randint(2, size=self.N_t)
            F = np.diag(f)
        else:    
            F = np.diag(f)
        return F 
    def gen_R_star(self):
        R_star = np.zeros([self.K, 1], dtype=float)
        R_star = np.random.uniform(0.1, 2, self.K)
        return R_star
#===================================================================
#===================================================================   
Var = var(K, N_t)
s_C = Var.gen_s_c()
s_P = Var.gen_s_p()  
w_C = Var.gen_w_c()
w_P = Var.gen_w_p()
A = Var.gen_F()
R_star = Var.gen_R_star()
#===================================================================
#===================================================================  
class Eta:
    def __init__(self, AA):
        self.AA = AA
        
    def eta(self):
        nnz_elements = np.count_nonzero(self.AA)
        eta = (nnz_elements/N_t) * 100
        eta_round_Down = trunc(eta*N_t)
        return eta, eta_round_Down 
#===================================================================
#=================================================================== 
ETA = Eta(A)
eta, eta_round_down  = ETA.eta()
#===================================================================
#=================================================================== 
class I_Dc:
    def __init__(self, etA, etA_round_down ):
        self.etA = etA
        self.etA_round_down  = etA_round_down 
     
    def I_dc(self):
        I_dc = I_L + (self.etA*N_t*(I_zero-I_L))/(self.etA_round_down+1)
        return I_dc
#===================================================================
#=================================================================== 
i_dc = I_Dc(eta, eta_round_down)   
I_DC = i_dc.I_dc() 
#===================================================================
#=================================================================== 
class gain_of_optical:
    def __init__(self, si, num_users, num_LEDs):
         self.si = si
         self.num_users = num_users
         self.num_LEDs = num_LEDs
         
    def G_VLC(self):
        g_vlc = np.zeros([self.num_users, self.num_LEDs])
        for i in range(self.num_users):
            for j in range(self.num_LEDs):
                if self.si[i,j] < (np.pi/3):
                    g_vlc[i,j] = (n_R**2)/(np.sin(np.pi/3) * np.sin(np.pi/3))
                else:
                    g_vlc[i,j] = 0
        return g_vlc     
#===================================================================
#=================================================================== 
G_opt = gain_of_optical(Si, K, N_t) 
G_vlc = G_opt.G_VLC()      
#===================================================================
#=================================================================== 
class channel:
    def __init__(self, g_VLC, num_users, num_LEDs, si, distance_led_pd):
      self.g_VLC = g_VLC
      self.num_users = num_users
      self.num_LEDs = num_LEDs
      self.si = si
      self.d = distance_led_pd

    def Channel(self):
      H = np.zeros([self.num_users, self.num_LEDs])
      for i in range(self.num_users):
        for j in range(self.num_LEDs):
            if self.si[i,j] < (np.pi/3):
                H[i,j] = ((m+1)*A_VLC*(self.g_VLC[i,j])*(np.cos(self.si[i,j])**(m))*(np.cos(self.si[i,j])))/(2*np.pi*self.d[i,j]*self.d[i,j])
            else:
                H[i,j] = 0
      return H
#===================================================================
#=================================================================== 
CH = channel(G_vlc, K, N_t, Si, distance_led_pd)   
H = CH.Channel()
#===================================================================
#===================================================================              
class rate:
    def __init__(self, H, A, w_c, w_p, num_users,n_t):
        self.H = H
        self.A = A
        self.w_c = w_c
        self.w_p = w_p
        self.num_users = num_users
        self.n_t = n_t
        
    def gamma_private(self):
        gamma_p = np.zeros(self.num_users)
        rate_p = np.zeros(self.num_users)
        for k in range(self.num_users):
            num = 0
            #num = np.linalg.norm(np.reshape(1*self.H[k,:], (1, self.n_t)) @ self.A * self.w_p[:,k])**2
            num = np.power(np.linalg.norm(np.reshape(1*self.H[k,:], (1, self.n_t)) @ self.A * self.w_p[:,k]),2)
            den1 = 0
            for i in range(self.num_users):
                if i != k:
                    den1 += np.linalg.norm(np.reshape(1*self.H[k,:], (1, self.n_t)) @ self.A * self.w_p[:,i])**2
                    
            gamma_p[k] = num / (den1 + sig2_k)   
            rate_p[k] = np.log2(1 + gamma_p[k])
        return gamma_p  , rate_p  
     
    def gamma_common(self):
        gamma_c = np.zeros(self.num_users)
        rate_c = np.zeros(self.num_users)
        for k in range(self.num_users):
            num = 0
            num = np.linalg.norm(np.reshape(1*self.H[k,:], (1, self.n_t)) @ self.A * self.w_c)**2
            den1 = 0
            for i in range(self.num_users):
                den1 += np.linalg.norm(np.reshape(1*self.H[k,:], (1, self.n_t)) @ self.A * self.w_p[:,i])**2
                    
            gamma_c[k] = num / (den1 + sig2_k)  
            rate_c[k] = np.log2(1+gamma_c[k])
        return gamma_c, rate_c        
#================================================================================
#================================================================================
Rate = rate(H, A, w_C, w_P, K, N_t)  
gamma_pri, rate_pri = Rate.gamma_private()
gamma_com, rate_com = Rate.gamma_common()
#================================================================================
#================================================================================
class sum_rate:
    def __init__(self, Rate_pri, Rate_com, num_users):
        self.Rate_pri = Rate_pri
        self.Rate_com = Rate_com
        self.num_users = num_users
        
    def Sum_Rate(self):
        Rate_sum = 0
        for i in range(self.num_users):
            Rate_sum += self.Rate_pri[i] + self.Rate_com[i]
        return Rate_sum  
#================================================================================
#================================================================================
SUM_rate = sum_rate(rate_pri, rate_com, K)
summ_rate = SUM_rate.Sum_Rate()
#================================================================================
#================================================================================
class harvested_power:
    def __init__(self, A, H, num_users, num_LEDs, i_DC):
        self.A = A
        self.H = H
        self.num_users = num_users
        self.num_LEDs = num_LEDs
        self.i_DC = i_DC
        
    def Harv_Power(self):
        P_H = np.zeros([self.num_users, self.num_LEDs])
        a = np.diag(self.A)
        for i in range(self.num_users):
            for j in range(self.num_LEDs):
                P_H[i,j] = a[j]*tau*V_t*self.i_DC*self.H[i,j]*np.log(1+((self.H[i,j]*self.i_DC)/I_dark_saturation))
        return P_H
#================================================================================
#================================================================================
Har_P = harvested_power(A, H, K, N_t, I_DC)
P_Har = Har_P.Harv_Power()
#================================================================================
#================================================================================   
class P_harvested_every_user:
    def __init__(self, P_Harvested, num_users):
        self.P_Harvested = P_Harvested
        self.num_users = num_users
        
    def harvested_power_every_PD(self):
        P_PD = np.zeros([self.num_users])
        for k in range(self.num_users):
            P_PD[k] = np.mean(self.P_Harvested[k])
        return P_PD
#================================================================================
#================================================================================
P_Harvested = P_harvested_every_user(P_Har, K)
PD_Power_harvested = P_Harvested.harvested_power_every_PD()
#================================================================================
#================================================================================
class P_total:
    def __init__(self, A, w_c, w_p, Na, i_DC, harvested_power_pd, num_users, num_LEDs):
        self.A = A
        self.w_c = w_c 
        self.w_p = w_p
        self.Na = Na
        self.i_DC = i_DC
        self.harvested_power_pd = harvested_power_pd
        self.num_users = num_users
        self.num_LEDs = num_LEDs
        
    def Total_power(self):
        a = np.diag(self.A)
        term1 = 0
        for i in range(self.num_LEDs):
            ww = np.zeros([self.num_LEDs])
            ww[i] = np.sum(self.w_p[i,:])
        for j in range(self.num_LEDs):
            term1 += zeta * a[j] * (self.w_c[j] + ww[j])
        term2 = 0
        for k in range(self.num_users):
            term2 += np.sum(self.harvested_power_pd)
        term3 = 0
        term3 = term1 + (phi_conversion_factor*self.Na*self.i_DC) - term2
        return term3
#================================================================================
#================================================================================
P_TOTAL = P_total(A, w_C, w_P, eta_round_down, I_DC, PD_Power_harvested, K, N_t)  
p_Total = P_TOTAL.Total_power()
#================================================================================
#================================================================================
class Reward:
    def __init__(self, rate_total):
        self.rate_total = rate_total
        
    def reward(self):
         obj = self.rate_total
         return obj
#================================================================================
#================================================================================
r = Reward(summ_rate)     
reward = r.reward()
#================================================================================
#================================================================================
def reset():
    episode = 0
    users_loc = Loc_user() 
    zaviaha = si_fi(users_loc, LEDs_loc, K, N_t) 
    distance_led_pd, Si = zaviaha.si()  
    Var = var(K, N_t)
    s_C = Var.gen_s_c()
    s_P = Var.gen_s_p()  
    w_C = Var.gen_w_c()
    w_P = Var.gen_w_p()
    A = Var.gen_F()
    R_star = Var.gen_R_star()
    ETA = Eta(A)
    eta, eta_round_down  = ETA.eta()
    i_dc = I_Dc(eta, eta_round_down)   
    I_DC = i_dc.I_dc() 
    G_opt = gain_of_optical(Si, K, N_t) 
    G_vlc = G_opt.G_VLC() 
    CH = channel(G_vlc, K, N_t, Si, distance_led_pd)   
    H = CH.Channel()
    Rate = rate(H, A, w_C, w_P, K, N_t)  
    gamma_pri, rate_pri = Rate.gamma_private()
    gamma_com, rate_com = Rate.gamma_common()
    SUM_rate = sum_rate(rate_pri, rate_com, K)
    summ_rate = SUM_rate.Sum_Rate()
    Har_P = harvested_power(A, H, K, N_t, I_DC)
    P_Har = Har_P.Harv_Power()
    P_Harvested = P_harvested_every_user(P_Har, K)
    PD_Power_harvested = P_Harvested.harvested_power_every_PD()
    P_TOTAL = P_total(A, w_C, w_P, eta_round_down, I_DC, PD_Power_harvested, K, N_t)  
    p_Total = P_TOTAL.Total_power()
    r = Reward(summ_rate)     
    reward = r.reward()
    r = reward.reshape(1,1)
    initial_action_w_C = np.hstack(w_C).reshape(1,-1)
    initial_action_w_P = np.hstack(w_P).reshape(1,-1)
    initial_action_A = np.hstack(np.diag(A)).reshape(1,N_t)
    initial_action_R_star = np.hstack(R_star).reshape(1,K)
    initial_action = np.hstack((initial_action_w_C, initial_action_w_P, initial_action_A, initial_action_R_star))
    a = (np.diag(w_P.T@w_P)).reshape(1, -1) ** 2
    if np.sum(a) == 0 : 
        b = 1
    else :
        b = np.sum(a)
    power_t = a / b
    a1 = (np.diag(w_C.T@w_C)).reshape(1, -1) ** 2
    if np.sum(a1) == 0 : 
        b1 = 1
    else :
        b1 = np.sum(a1)
    power_t1 = a1 / b1
    s1 = (rate_pri.reshape(1,-1)) #/(np.max(rate_pri)+1)
    s2 = (rate_com.reshape(1,-1)) #/(np.max(rate_com)+1)
    s3 = (PD_Power_harvested.reshape(1,-1)) #/(np.max(PD_Power_harvested)+1)
    state = np.hstack((power_t, power_t1, s1, s2, s3))
    return state
#==================================================================================
#==================================================================================
def step(action_PPO):
    action_PPO = action_PPO.reshape(Action_size_PPO,1)
    start = 0
    end = K*N_t
    w_p_vec = action_PPO[start:end]
    w_p_matrix = w_p_vec.reshape(N_t, K)
    
    start = end
    end = end + N_t
    w_c_vec = action_PPO[start:end]
    w_c_matrix = w_c_vec.reshape(N_t,1)
    
    A_vec = np.array([1,1,1,0,0,0])
    A_vec = A_vec.reshape(N_t,) 
    A_matrix = np.diag(A_vec)
    
    start = end
    end = end + K
    r_star = ((action_PPO[start:end]+1)/2)
    r_star = r_star.reshape(K,)
    
    a = (np.diag(w_p_matrix.T@w_p_matrix)).reshape(1, -1) ** 2
    if np.sum(a) == 0 : 
        b = 1
    else :
        b = np.sum(a)
    power_t = a / b
    a1 = (np.diag(w_c_matrix.T@w_c_matrix)).reshape(1, -1) ** 2
    if np.sum(a1) == 0 : 
        b1 = 1
    else :
        b1 = np.sum(a1)
    power_t1 = a1 / b1
    
    ETA = Eta(A_matrix)
    eta, eta_round_down  = ETA.eta()
    i_dc = I_Dc(eta, eta_round_down)   
    I_DC = i_dc.I_dc() 
    G_opt = gain_of_optical(Si, K, N_t) 
    G_vlc = G_opt.G_VLC() 
    CH = channel(G_vlc, K, N_t, Si, distance_led_pd)   
    H = CH.Channel()
    Rate = rate(H, A_matrix, w_c_matrix, w_p_matrix, K, N_t)  
    gamma_pri, rate_pri = Rate.gamma_private()
    gamma_com, rate_com = Rate.gamma_common()
    SUM_rate = sum_rate(rate_pri, rate_com, K)
    summ_rate = SUM_rate.Sum_Rate()
    Har_P = harvested_power(A_matrix, H, K, N_t, I_DC)
    P_Har = Har_P.Harv_Power()
    P_Harvested = P_harvested_every_user(P_Har, K)
    PD_Power_harvested = P_Harvested.harvested_power_every_PD()
    P_TOTAL = P_total(A_matrix, w_c_matrix, w_p_matrix, eta_round_down, I_DC, PD_Power_harvested, K, N_t)  
    p_Total = P_TOTAL.Total_power()
    r = Reward(summ_rate)     
    reward = r.reward()
    r = reward.reshape(1,1)
    initial_action_w_C = np.hstack(w_c_matrix).reshape(1,-1)
    initial_action_w_P = np.hstack(w_p_matrix).reshape(1,-1)
    initial_action_A = np.hstack(np.diag(A_matrix)).reshape(1,N_t)
    initial_action_R_star = np.hstack(r_star).reshape(1,K)
    initial_action = np.hstack((initial_action_w_C, initial_action_w_P, initial_action_A, initial_action_R_star))
    s1 = (rate_pri.reshape(1,-1)) #/np.max(rate_pri)
    s2 = (rate_com.reshape(1,-1)) #/np.max(rate_com)
    s3 = (PD_Power_harvested.reshape(1,-1)) #/np.max(PD_Power_harvested)
    next_state = np.hstack((power_t, power_t1, s1, s2, s3))
    return next_state, rate_com, r_star, rate_pri, p_Total, PD_Power_harvested, w_c_matrix, w_p_matrix, I_DC, r
action_PPO = np.random.randn(1, Action_size_PPO)
next_state, rate_com, r_star, rate_pri, p_Total, PD_Power_harvested, w_c_matrix, w_p_matrix, I_DC, reward = step(action_PPO)
#===============================================================
#===============================================================
def c1(rate_com, r_star):
    check_c1 = 0
    if np.min(rate_com) > np.sum(r_star):
        check_c1 = 1
    else:
        check_c1 = 0
    return check_c1    
#==================================
check_c1 = c1(rate_com, r_star)
#================================================================
#================================================================
def c2(rate_pri, r_star):
    check_c2 = 0
    for i in range(K):
        if (rate_pri[i] + r_star[i]) > QoS[i]:
            check_c2 += 1
        else:
            check_c2 += 0
    if check_c2 == K:
      check_c2 = 1
    else:
      check_c2 = 0
    return check_c2 
#=================================================================
check_c2 =  c2(rate_pri, r_star)
#=================================================================
#=================================================================
def c3(p_Total):
    check_c3 = 0
    if p_Total<P_max:
        check_c3 = 1
    else:
        check_c3 = 0
    return check_c3
#============================================
check_c3 = c3(p_Total)  
#=====================================================================
#=====================================================================
def c4(PD_Power_harvested):
    check_c4 = 0
    for i in range(K):
        if PD_Power_harvested[i] > Har_Pow_PD[i]:
            check_c4 += 1
        else:
            check_c4 += 0
    if check_c4 == K:
       check_c4 = 1
    else:
       check_c4 = 0
    return check_c4  
#=============================================
check_c4 = c4(PD_Power_harvested)
#======================================================================
#======================================================================
def c5(w_c_matrix, w_p_matrix, I_DC):
    check_c5 = 0
    for i in range(N_t):
        a = np.array([I_DC-I_L, I_DC - I_h])
        if (np.abs(w_c_matrix[i]) + np.abs(np.sum(w_p_matrix[i,:]))) < np.min(a):
            check_c5 += 1
        else:
            check_c5 += 0
    if check_c5 == N_t:
       check_c5 = 1
    else:
       check_c5 = 0
    return check_c5
#===================================
check_c5 =   c5(w_c_matrix, w_p_matrix, I_DC)      

def compute_discounted_return(rewards, last_value):
    returns = np.zeros_like(rewards)
    n_step = len(rewards)

    for t in reversed(range(n_step)):
        if t == n_step - 1:
            returns[t] = rewards[t] + gamma * last_value
        else:
            returns[t] = rewards[t] + gamma * returns[t+1]

    return returns
    
#Let's go----------------------------------------------------------------------------------------------------
n_episode = 30000
n_step= 250

reward_episode = np.zeros([n_episode])
objective_episode = np.zeros([n_episode])

policy_net = PolicyNet(state_size, Action_size_PPO)
value_net = ValueNet(state_size)
agent = PPO(policy_net, value_net)

done = False

for i_episode in range (n_episode):
    state = reset()
    state = state/np.max(state)
    reward_step = np.zeros([n_step])
    objective_step = np.zeros([n_step])
#-------------------------
    mb_states = np.zeros((n_step, state_size), dtype=np.float32)
    mb_actions = np.zeros((n_step, Action_size_PPO), dtype=np.float32)
    mb_values = np.zeros((n_step,), dtype=np.float32)
    mb_rewards = np.zeros((n_step,), dtype=np.float32)
    mb_a_logps = np.zeros((n_step,), dtype=np.float32)
#-----------------------
    for i_step in range (n_step):     
        mb_states[i_step] = state
        state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device='cpu')
        action_tensor, a_logp = policy_net(state_tensor)
        value = value_net(state_tensor)
        action = action_tensor.cpu().numpy()[0]
        a_logp = a_logp.detach().numpy()
        value  = value.detach().numpy()
        mb_actions[i_step] = action
        mb_a_logps[i_step] = a_logp
        mb_values[i_step] = value
        next_state, rate_com, r_star, rate_pri, p_Total, PD_Power_harvested, w_c_matrix, w_p_matrix, I_DC, reward = step(action)
        next_state = next_state/np.max(next_state)
        #Check Constraint---------------------------------------------------------------------------
        check_c1 = c1(rate_com, r_star)
        check_c2 =  c2(rate_pri, r_star)
        check_c3 = c3(p_Total)  
        check_c4 = c4(PD_Power_harvested)
        check_c5 = c5(w_c_matrix, w_p_matrix, I_DC)      
        print('C1:',check_c1,'C2:',check_c2,'C3:',check_c3,'C4:',check_c4,'C5:',check_c5)
        reward_final = 0
        if check_c1 == 1 :
            reward_final += reward
        if check_c2 == 1 :
            reward_final += reward
        if check_c3 == 1 :
            reward_final += reward
        if check_c4 == 1 :
            reward_final += reward
        if check_c5 == 1 :
            reward_final += reward
            
        reward_step[i_step] = reward_final
        objective_step[i_step] = reward
        state = next_state
        print('episode:',i_episode,' step:',i_step, 'reward', reward_final)
        
    last_value = value_net(state_tensor)
    returns = compute_discounted_return(reward_step, last_value)
    advs = returns - mb_values
    advs = (advs - advs.mean()) / (advs.std() + 1e-6)
    state = state.reshape(state_size)
    action = action.reshape(Action_size_PPO)
    
    pg_loss, v_loss, ent = agent.train(mb_states, mb_actions, mb_values, advs, returns, mb_a_logps)
    reward_episode[i_episode] = np.mean(reward_step[:])
    objective_episode[i_episode] = np.mean(reward_step[:])
    
plt.plot(reward_episode)
plt.show()
plt.plot(objective_episode)
print('finished')
    
  
            
                
        
    