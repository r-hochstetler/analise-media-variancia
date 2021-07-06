#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de análise média-variância da matriz elétrica

Instituto Acende Brasil
P&D Matriz Robusta
dezembro/2020

Pressupostos de custos das usinas especificados nas planilhas 
    - custo de geração termelétrica 
    - MDI_Dados Expansão

"""

# bibliotecas utilizadas
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import requests
#import datetime
#import matplotlib
#matplotlib.style.use('seaborn')

# diretorio de trabalho

diretorio = os.getcwd()
print()
print()
print('DIRETÓRIO ATUAL')
print(diretorio)
pd.options.display.float_format = '{:,.1f}'.format
print()
print('AQUISIÇÃO DE DADOS')


def plotar_graficos_eol(df_var_eol):
    fig, ax = plt.subplots(4, 1, figsize=(12, 16))
    ax[0].set_title('geração eólica')
    ax[0].plot (df_var_eol['g_eol'], color='lightblue')
    ax[0].set_ylabel('MWm')
    ax[1].set_title('variação percentual da geração eólica')
    ax[1].plot (df_var_eol['var_%_g_eol'], color='lightblue')
    ax[1].set_ylabel('%')
    ax[2].set_title('custo marginal de operação')
    ax[2].plot(df_var_eol['cmo'], color='black')
    ax[2].set_ylabel('R$/MWh')
    ax[3].set_title('volatilidade do custo eólico')
    ax[3].plot(df_var_eol['var_perc_c_eol'], color='lightblue')
    ax[3].set_ylabel('%')
    plt.show()

def plotar_graficos_sol(df_var_sol):
    fig, ax = plt.subplots(4, 1, figsize=(12, 16))
    ax[0].set_title('geração solar')
    ax[0].plot (df_var_sol['g_sol'], color='gold')
    ax[0].set_ylabel('MWm')
    ax[1].set_title('variação percentual da geração solar')
    ax[1].plot (df_var_sol['var_%_g_sol'], color='gold')
    ax[1].set_ylabel('%')
    ax[2].set_title('custo marginal de operação')
    ax[2].plot(df_var_sol['cmo'], color='black')
    ax[2].set_ylabel('R$/MWh')
    ax[3].set_title('volatilidade do custo solar')
    ax[3].plot(df_var_sol['var_perc_c_sol'], color='gold')
    ax[3].set_ylabel('%')
    plt.show()

def plotar_graficos_hidro(df_var_hidro):
    fig, ax = plt.subplots(4, 1, figsize=(12, 16))
    ax[0].set_title('geração hidrelétrica')
    ax[0].plot (df_var_hidro['g_hidro'], color='navy')
    ax[0].set_ylabel('MWm')
    ax[1].set_title('variação percentual da geração hidrelétrica')
    ax[1].plot (df_var_hidro['var_%_g_hidro'], color='navy')
    ax[1].set_ylabel('%')
    ax[2].set_title('custo marginal de operação')
    ax[2].plot(df_var_hidro['cmo'], color='black')
    ax[2].set_ylabel('R$/MWh')
    ax[3].set_title('volatilidade do custo hidrelétrico')
    ax[3].plot(df_var_hidro['var_perc_c_hidro'], color='navy')
    ax[3].set_ylabel('%')
    plt.show()


# CUSTO MARGINAL DE OPERAÇÃO
# obtenção de dados de CMO - fonte ONS
arquivo = 'cmo.csv'
df_cmo = pd.read_csv(os.path.normcase(os.path.join(diretorio, arquivo)), engine='python', sep=';')
df_cmo['data_horario']=pd.to_datetime(df_cmo['data_horario'], format='%d/%m/%Y %H:%M')
df_cmo.set_index('data_horario', inplace=True)
df_cmo['cmo'] = pd.to_numeric(df_cmo['cmo'], downcast='float')
#df_cmo.plot() 
plt.title('custo marginal de operação', fontsize=14)
plt.plot(df_cmo['cmo'])
plt.ylabel('R$/MWh')
plt.show()


# GERAÇÃO HIDRELÉTRICA

# obtenção e processamento de geração hidrelétrica - fonte ONS
arquivo = 'g_hidro.csv'
df_hidro = pd.read_csv(os.path.normcase(os.path.join(diretorio, arquivo)), engine='python', sep=';')
df_hidro['data_horario'] = pd.to_datetime(df_hidro['data_horario'], format='%m/%d/%Y %H:%M')
df_hidro.set_index('data_horario', inplace=True)
df_hidro.plot()
g_hidro_med = df_hidro['g_hidro'].mean()
df_hidro['g_hidro_12m'] = df_hidro['g_hidro'].rolling(12 * 1).mean()
df_hidro['g_hidro_12m'].plot()
df_hidro['g_hidro_24m'] = df_hidro['g_hidro'].rolling(12 * 2).mean()
df_hidro['g_hidro_24m'].plot()
df_hidro['g_hidro_36m'] = df_hidro['g_hidro'].rolling(12 * 3).mean()
df_hidro['g_hidro_36m'].plot()
plt.show()
df_hidro.drop(columns=['g_hidro_12m', 'g_hidro_24m', 'g_hidro_36m'], inplace=True)
c_med_hidro = 319   # custo com base no MDI do PDE 2029
df_var_hidro = pd.concat([df_cmo, df_hidro], join='outer', axis=1, sort=True)
#del df_hidro
for i in range(0, df_var_hidro.shape[0]):
    if np.isnan(df_var_hidro.iloc[i, 1]):
        df_var_hidro.iloc[i, 1] = df_var_hidro.iloc[i - 1, 1]
df_var_hidro['g_hidro_12m'] = df_var_hidro['g_hidro'].rolling(1 * 360 * 24).mean()
df_var_hidro = df_var_hidro.dropna()
df_var_hidro['var_%_g_hidro'] = df_var_hidro.apply(
        lambda row: (row.g_hidro_12m - g_hidro_med) / g_hidro_med, axis=1)
df_var_hidro['var_c_hidro'] = df_var_hidro.apply(
        lambda row: -(row.g_hidro_12m - g_hidro_med) * row.cmo, axis=1)
df_var_hidro['var_perc_c_hidro'] = df_var_hidro.apply(
        lambda row: row.var_c_hidro / (c_med_hidro * g_hidro_med), axis=1)
df_var_hidro['c_hidro'] = df_var_hidro.apply(
        lambda row: c_med_hidro * (1 + row.var_perc_c_hidro), axis=1)
plotar_graficos_hidro(df_var_hidro)
for col in df_var_hidro.columns:
    print(col)
    
    
# GERAÇÃO EÓLICA
# obtenção e processamento de dados de geração eólica - fonte ONS
arquivo = 'g_eol.csv'
df_eol = pd.read_csv(os.path.normcase(os.path.join(diretorio, arquivo)), engine='python', sep=';')
df_eol['data_horario'] = pd.to_datetime(df_eol['data_horario'], format='%d/%m/%Y %H:%M')
df_eol.set_index('data_horario', inplace=True)
df_eol.plot()
#c_med_eol = 160    # custo com base nos leilões
c_med_eol = 199    # custo com base no PDE 2029
df_var_eol = pd.concat([df_cmo, df_eol], axis=1, join='inner', sort=True)
df_var_eol['g_eol_med'] = df_var_eol['g_eol'].rolling(365 * 24).mean()
df_var_eol = df_var_eol.dropna()
df_var_eol['var_%_g_eol'] = df_var_eol.apply(
        lambda row: (row.g_eol - row.g_eol_med) / row.g_eol_med, axis=1)
df_var_eol['var_c_eol'] = df_var_eol.apply(
        lambda row: -(row.g_eol - row.g_eol_med) * row.cmo, axis=1)
df_var_eol['var_perc_c_eol'] = df_var_eol.apply(
        lambda row: row.var_c_eol / (c_med_eol * row.g_eol_med), axis=1)
plotar_graficos_eol(df_var_eol)

# seleção de segemento de dados estável
df_eol['g_eol_med']= df_eol.rolling(365 * 24).mean()
del df_var_eol
df_var_eol = pd.concat([df_cmo, df_eol], axis=1, join='outer', sort=True)
df_corte = df_var_eol.loc[df_var_eol.index >= '2015-01-01 00:00:00']
n_linhas_corte = df_corte.shape[0]
n_linhas_df_var_eol = df_var_eol.shape[0]
for i in range(n_linhas_df_var_eol - n_linhas_corte -1, -1, -1):
    j = i + n_linhas_corte
    df_var_eol.iloc[i, 1] = df_var_eol.iloc[j, 1] 
    df_var_eol.iloc[i, 2] = df_var_eol.iloc[j, 2] 
df_var_eol['var_%_g_eol'] = df_var_eol.apply(
        lambda row: (row.g_eol - row.g_eol_med) / row.g_eol_med, axis=1)
df_var_eol['var_c_eol'] = df_var_eol.apply(
        lambda row: -(row.g_eol - row.g_eol_med) * row.cmo, axis=1)
df_var_eol['var_perc_c_eol'] = df_var_eol.apply(
        lambda row: row.var_c_eol / (c_med_eol * row.g_eol_med), axis=1)
df_var_eol['c_eol'] = df_var_eol.apply(
        lambda row: c_med_eol * (1 + row.var_perc_c_eol), axis=1)
plotar_graficos_eol(df_var_eol)
for col in df_var_eol.columns:
    print(col)


# GERAÇÃO SOLAR
# obtenção e processamento de dados de geração solar - fonte ONS
arquivo = 'g_sol.csv'
df_sol = pd.read_csv(os.path.normcase(os.path.join(diretorio, arquivo)), engine='python', sep=';')
df_sol['data_horario'] = pd.to_datetime(df_sol['data_horario'], format='%d/%m/%Y %H:%M')
df_sol.set_index('data_horario', inplace=True)
df_sol.plot()
#c_med_sol = 150    # custo com base nos leilões
c_med_sol = 200    # custo com base no PDE 2029 (com desconto)
df_var_sol = pd.concat([df_cmo, df_sol], axis=1, join='inner', sort=True)
df_var_sol['g_sol_med'] = df_var_sol['g_sol'].rolling(365 * 24).mean()
df_var_sol = df_var_sol.dropna()
df_var_sol['var_%_g_sol'] = df_var_sol.apply(
        lambda row: (row.g_sol - row.g_sol_med) / row.g_sol_med, axis=1)
df_var_sol['var_c_sol'] = df_var_sol.apply(
        lambda row: -(row.g_sol - row.g_sol_med) * row.cmo, axis=1)
df_var_sol['var_perc_c_sol'] = df_var_sol.apply(
        lambda row: row.var_c_sol / (c_med_sol * row.g_sol_med), axis=1)
plotar_graficos_sol(df_var_sol)

# seleção de segemento de dados estável
df_sol['g_sol_med']= df_sol.rolling(360 * 24).mean()
del df_var_sol
df_var_sol = pd.concat([df_cmo, df_sol], axis=1, join='outer', sort=True)
df_corte = df_var_sol.loc[df_var_sol.index >= '2019-01-01 00:00:00']
n_linhas_corte = df_corte.shape[0]
n_linhas_df_var_sol = df_var_sol.shape[0]
for i in range(n_linhas_df_var_sol - n_linhas_corte -1, -1, -1):
    j = i + n_linhas_corte
    df_var_sol.iloc[i, 1] = df_var_sol.iloc[j, 1] 
    df_var_sol.iloc[i, 2] = df_var_sol.iloc[j, 2] 
df_var_sol['var_%_g_sol'] = df_var_sol.apply(
        lambda row: (row.g_sol - row.g_sol_med) / row.g_sol_med, axis=1)
df_var_sol['var_c_sol'] = df_var_sol.apply(
        lambda row: -(row.g_sol - row.g_sol_med) * row.cmo, axis=1)
df_var_sol['var_perc_c_sol'] = df_var_sol.apply(
        lambda row: row.var_c_sol / (c_med_sol * row.g_sol_med), axis=1)
df_var_sol['c_sol'] = df_var_sol.apply(
        lambda row:  c_med_sol * (1 + row.var_perc_c_sol), axis=1)
plotar_graficos_sol(df_var_sol)
for col in df_var_sol.columns:
    print(col)


# GERAÇÃO TERMELÉTRICA
# obtenção de dados de custo de combustíveis - fonte Banco Mundial
arquivo = 'p_comb.csv'
df_comb = pd.read_csv(os.path.normcase(os.path.join(diretorio, arquivo)), engine='python', sep=';')
df_comb['data_horario'] = pd.to_datetime(df_comb['data_horario'], format='%d/%m/%Y %H:%M')
df_comb.set_index('data_horario', inplace=True)

for col in df_comb.columns:
    print(col)

#fig, ax = plt.subplots(5, 1, figsize=(12, 20))
#ax[0].set_title('preço do petróleo (Brent)')
#ax[0].plot (df_comb['p_petroleo'], color='grey')
#ax[0].set_ylabel('US$ / barril')
#ax[1].set_title('preço do carvão (australiano)')
#ax[1].plot (df_comb['p_carvao_australiano'], color='black')
#ax[1].set_ylabel('US$ / tonelada')
#ax[2].set_title('preço do gás natural (Henry Hub)')
#ax[2].plot (df_comb['p_GN_HH'], color='gold')
#ax[2].set_ylabel('US$ / milhão de BTU')
#ax[3].set_title('preço do gás natural (Europa)')
#ax[3].plot (df_comb['p_GN_Europa'], color='goldenrod')
#ax[3].set_ylabel('US$ / milhão de BTU')
#ax[4].set_title('preço do gás natural liquefeito (Japão)')
#ax[4].plot (df_comb['p_GNL'], color='darkgoldenrod')
#ax[4].set_ylabel('US$ / milhão de BTU')
#plt.show()

df_var_termo = pd.concat([df_cmo, df_comb], axis=1, sort=True)
#del df_comb
for i in range(0, df_var_termo.shape[0]):
    if np.isnan(df_var_termo.iloc[i,-1]):
        df_var_termo.iloc[i,-1] = df_var_termo.iloc[i - 1,-1]
    if np.isnan(df_var_termo.iloc[i,-2]):
        df_var_termo.iloc[i,-2] = df_var_termo.iloc[i - 1,-2]
    if np.isnan(df_var_termo.iloc[i,-3]):
        df_var_termo.iloc[i,-3] = df_var_termo.iloc[i - 1,-3]
    if np.isnan(df_var_termo.iloc[i,-4]):
        df_var_termo.iloc[i,-4] = df_var_termo.iloc[i - 1,-4]
    if np.isnan(df_var_termo.iloc[i,-5]):
        df_var_termo.iloc[i,-5] = df_var_termo.iloc[i - 1,-5]
    if np.isnan(df_var_termo.iloc[i,-6]):
        df_var_termo.iloc[i,-6] = df_var_termo.iloc[i - 1,-6]
df_var_termo = df_var_termo.dropna()

p_petr_med = df_var_termo['p_petroleo'].mean()
p_carvao_med = df_var_termo['p_carvao_australiano'].mean()
p_GN_HH_med = df_var_termo['p_GN_HH'].mean()
p_GN_Europa_med = df_var_termo['p_GN_Europa'].mean()
p_GNL_med = df_var_termo['p_GNL'].mean()
df_var_termo['var_%_petr'] = df_var_termo.apply(
        lambda row: (row.p_petroleo - p_petr_med) / p_petr_med, axis=1)
df_var_termo['var_%_carvao'] = df_var_termo.apply(
        lambda row: (row.p_carvao_australiano - p_carvao_med) / p_carvao_med, axis=1)
df_var_termo['var_%_GN_HH'] = df_var_termo.apply(
        lambda row: (row.p_GN_HH - p_GN_HH_med) / p_GN_HH_med, axis=1)
df_var_termo['var_%_GN_Europa'] = df_var_termo.apply(
        lambda row: (row.p_GN_Europa - p_GN_Europa_med) / p_GN_Europa_med, axis=1)
df_var_termo['var_%_GNL'] = df_var_termo.apply(
        lambda row: (row.p_GNL - p_GNL_med) / p_GNL_med, axis=1)

plt.figure(figsize=(12,4))
plt.title('volatilidade de preços dos combustíveis')
plt.plot(df_var_termo['var_%_petr'], color='grey', linewidth=2.5, label='petróleo (Brent)')
plt.plot(df_var_termo['var_%_carvao'], color='black', label='carvão (Austrália)')
plt.plot(df_var_termo['var_%_GN_HH'], color='gold', label='gás natural (Henry Hub)')
plt.plot(df_var_termo['var_%_GN_Europa'], color='goldenrod', label='gás natural (Europa)')
plt.plot(df_var_termo['var_%_GNL'], color='darkgoldenrod', label='gás natural liquefeito (Japão)')
plt. ylabel('%')
plt.legend()
plt.show()

# custo fixo em função da tecnologia
c_f_ute_gn_cs = 87
c_f_ute_gn_cc = 94
c_f_ute_carvao = 184
c_f_ute_gn_renov = 73
c_f_ute_carvao_renov = 116

# preco spot atual de cada combustível dada a forma de contratação (MDI do PDE 2029)
gn_0 = 350          # ciclo aberto / mercado spot
gn_50 = 300         # ciclo fechado / desconto pelo take-or-pay 50%
gn_90 = 270         # ciclo fechado / desconto pelo take-or-pay 90%
carvao = 120        # pressupõe take-or-pay de 50%
gn_renov = 300      # usinas em fim de contrato 
carvao_renov = 300  # usinas em fim de contrato 
# fator de capacidade esperdado dado a inflexibilidade contratada
fc_0 = .30       
fc_50 = .60
fc_90 = .95

# Evolução do custo unitário total teremelétrico (ICB = CF + COP + CEC)
# CF - custo fixo anualizado por MWm
# COP - custo variável unitário de operação 
# CEC - custo variável de aquisição de energia no mercado de curto prazo quando a usina não é despachada por mérito
cec = .5    # custo do CMO em relação ao seu próprio CVU

# custo do combustível atual
gn_europa =     [gn_0]
gn_hh =         [gn_0]
gn_japao =      [gn_0]
gn_brent_0 =    [gn_0]
gn_brent_50 =   [gn_50]
gn_brent_90 =   [gn_90]
carvao =        [carvao]
gn_renov =      [gn_renov]
carvao_renov =  [carvao_renov]

# custo total atual
c_gn_europa =   [c_f_ute_gn_cs          + fc_0  * gn_europa[0]      + (1 - fc_0)  * cec * gn_europa[0]]
c_gn_hh =       [c_f_ute_gn_cs          + fc_0  * gn_hh[0]          + (1 - fc_0)  * cec * gn_hh[0]]
c_gn_japao =    [c_f_ute_gn_cs          + fc_0  * gn_japao[0]       + (1 - fc_0)  * cec * gn_japao[0]]
c_gn_brent_0 =  [c_f_ute_gn_cs          + fc_0  * gn_brent_0[0]     + (1 - fc_0)  * cec * gn_brent_0[0]]
c_gn_brent_50 = [c_f_ute_gn_cc          + fc_50 * gn_brent_50[0]    + (1 - fc_50) * cec * gn_brent_50[0]]
c_gn_brent_90 = [c_f_ute_gn_cc          + fc_90 * gn_brent_90[0]    + (1 - fc_90) * cec * gn_brent_90[0]]
c_carvao =      [c_f_ute_carvao         + fc_50 * carvao[0]         + (1 - fc_50) * cec * carvao[0]]
c_gn_renov =    [c_f_ute_gn_renov       + fc_0 * gn_renov[0]        + (1 - fc_0) * cec * gn_renov[0]]
c_carvao_renov =[c_f_ute_carvao_renov   + fc_0 * carvao_renov[0]    + (1 - fc_0) * cec * carvao_renov[0]]

for col in df_var_termo.columns:
    print(col)
    
for i in range(df_var_termo.shape[0] - 2, -1, -1):
    # variação do custo do combustível em função da indexação
    gn_brent_0.insert(  0, gn_brent_0[0]   * df_var_termo.iloc[i, 1] / df_var_termo.iloc[i + 1, 1]) # variação do preço do petróleo Brent
    gn_brent_50.insert( 0, gn_brent_50[0]  * df_var_termo.iloc[i, 1] / df_var_termo.iloc[i + 1, 1]) # variação do preço do petróleo Brent
    gn_brent_90.insert( 0, gn_brent_90[0]  * df_var_termo.iloc[i, 1] / df_var_termo.iloc[i + 1, 1]) # variação do preço do petróleo Brent
    carvao.insert(      0, carvao[0]       * df_var_termo.iloc[i, 2] / df_var_termo.iloc[i + 1, 2]) # variação do preço do carvão Austrália
    gn_hh.insert(       0, gn_hh[0]        * df_var_termo.iloc[i, 4] / df_var_termo.iloc[i + 1, 4]) # variação do preço GN Henry Hub
    gn_europa.insert(   0, gn_europa[0]    * df_var_termo.iloc[i, 5] / df_var_termo.iloc[i + 1, 5]) # variação do preço GN Europa
    gn_japao.insert(    0, gn_japao[0]     * df_var_termo.iloc[i, 6] / df_var_termo.iloc[i + 1, 6]) # variação do preço GNL Japão
    gn_renov.insert(    0, gn_renov[0]     * df_var_termo.iloc[i, 1] / df_var_termo.iloc[i + 1, 1]) # variação do preço do petróleo Brent
    carvao_renov.insert(0, carvao_renov[0] * df_var_termo.iloc[i, 2] / df_var_termo.iloc[i + 1, 2]) # variação do preço do carvão Austrália

    # variação do custo total
    c_gn_europa.insert(     0, c_f_ute_gn_cs        + fc_0 * gn_europa[0]            + (1 - fc_0)  * cec * gn_europa[0])
    c_gn_hh.insert(         0, c_f_ute_gn_cs        + fc_0 * gn_hh[0]                + (1 - fc_0)  * cec * gn_hh[0])
    c_gn_japao.insert(      0, c_f_ute_gn_cs        + fc_0 * gn_japao[0]             + (1 - fc_0)  * cec * gn_japao[0])
    c_gn_brent_0.insert(    0, c_f_ute_gn_cs        + fc_0 * gn_brent_0[0]           + (1 - fc_0)  * cec * gn_brent_0[0])
    c_gn_brent_50.insert(   0, c_f_ute_gn_cc        + (fc_50 - .50) * gn_brent_50[0] + (1 - fc_50) * cec * gn_brent_50[0]     + .50 * gn_brent_50[-1])
    c_gn_brent_90.insert(   0, c_f_ute_gn_cc        + (fc_90 - .90) * gn_brent_90[0] + (1 - fc_90) * cec * gn_brent_90[0]     + .90 * gn_brent_90[-1])
    c_carvao.insert(        0, c_f_ute_carvao       + (fc_50 - .50) * carvao[0]      + (1 - fc_50) * cec * carvao[0]          + .50 * carvao[-1])
    c_gn_renov.insert(      0, c_f_ute_gn_renov     + fc_0 * gn_renov[0]            + (1 - fc_0) * cec * gn_renov[0])
    c_carvao_renov.insert(  0, c_f_ute_carvao_renov + fc_0 * carvao_renov[0]        + (1 - fc_0) * cec * carvao_renov[0])

df_var_termo['c_gn_europa'] = c_gn_europa
df_var_termo['c_gn_hh'] = c_gn_hh
df_var_termo['c_gn_japao'] = c_gn_japao
df_var_termo['c_gn_brent_0'] = c_gn_brent_0
df_var_termo['c_gn_brent_50'] = c_gn_brent_50
df_var_termo['c_gn_brent_90'] = c_gn_brent_90
df_var_termo['c_carvao'] = c_carvao
df_var_termo['c_gn_renov'] = c_gn_renov
df_var_termo['c_carvao_renov'] = c_carvao_renov

c_gn_europa_med = df_var_termo['c_gn_europa'].mean()
c_gn_hh_med = df_var_termo['c_gn_hh'].mean()
c_gn_japao_med = df_var_termo['c_gn_japao'].mean()
c_gn_brent_0_med = df_var_termo['c_gn_brent_0'].mean()
c_gn_brent_50_med = df_var_termo['c_gn_brent_50'].mean()
c_gn_brent_90_med = df_var_termo['c_gn_brent_90'].mean()
c_carvao_med = df_var_termo['c_carvao'].mean() 
c_gn_renov_med = df_var_termo['c_gn_renov'].mean()
c_carvao_renov_med = df_var_termo['c_carvao_renov'].mean()

df_var_termo['var_%_gn_europa'] = df_var_termo.apply(
        lambda row: row.c_gn_europa / c_gn_europa_med - 1, axis = 1)
df_var_termo['var_%_gn_hh'] = df_var_termo.apply(
        lambda row: row.c_gn_hh / c_gn_hh_med - 1, axis = 1)
df_var_termo['var_%_gn_japao'] = df_var_termo.apply(
        lambda row: row.c_gn_japao / c_gn_japao_med - 1, axis = 1)
df_var_termo['var_%_gn_brent_0'] = df_var_termo.apply(
        lambda row: row.c_gn_brent_0 / c_gn_brent_0_med - 1, axis = 1)
df_var_termo['var_%_gn_brent_50'] = df_var_termo.apply(
        lambda row: row.c_gn_brent_50 / c_gn_brent_50_med - 1, axis = 1)
df_var_termo['var_%_gn_brent_90'] = df_var_termo.apply(
        lambda row: row.c_gn_brent_90 / c_gn_brent_90_med - 1, axis = 1)
df_var_termo['var_%_carvao'] = df_var_termo.apply(
        lambda row: row.c_carvao / c_carvao_med - 1, axis = 1)
df_var_termo['var_%_gn_renov'] = df_var_termo.apply(
        lambda row: row.c_gn_renov / c_gn_renov_med - 1, axis = 1)
df_var_termo['var_%_carvao_renov'] = df_var_termo.apply(
        lambda row: row.c_carvao_renov / c_carvao_renov_med - 1, axis = 1)

plt.figure(figsize=(12,6))
plt.title('volatilidade de custos de geração termelétrica')
plt.plot(df_var_termo['var_%_gn_europa'], color='darkgoldenrod', 
         label='GN flexível indexado aos preços da Europa')
plt.plot(df_var_termo['var_%_gn_hh'], color='goldenrod', 
         label='GN flexível indexado aos preços Henry Hub')
plt.plot(df_var_termo['var_%_gn_japao'], color='gold', 
         label='GN flexível indexado ao Japão')
plt.plot(df_var_termo['var_%_gn_brent_0'], color='grey', linewidth= 1.5,
         label='GN flexível indexado aos preços do Brent')
plt.plot(df_var_termo['var_%_gn_brent_50'], color='silver', linewidth= 1.5,
         label='GN com inflexibilidade de 50% indexado aos preços do Brent')
plt.plot(df_var_termo['var_%_gn_brent_90'], color='lightgrey', linewidth= 1.5, 
         label='GN com inflexibilidade de 90% indexado aos preços do Brent')
plt.plot(df_var_termo['var_%_gn_renov'], color='grey', linewidth= 1.5, 
         linestyle='dashed', label='GN renovado')
plt.plot(df_var_termo['var_%_carvao_renov'], color='black', linestyle='dashed',
         label='carvão renovação')
plt.plot(df_var_termo['var_%_carvao'], color='black', 
         label='carvão nacional com inflexibilidade de 50%')
plt. ylabel('%')
plt.legend()
plt.show()


custos_historicos = pd.concat([df_var_hidro['c_hidro'], 
                             df_var_eol['c_eol'], 
                             df_var_sol['c_sol'],
                             df_var_termo['c_carvao'],
                             df_var_termo['c_carvao_renov'],
                             df_var_termo['c_gn_brent_0'],
                             df_var_termo['c_gn_brent_50'], 
                             df_var_termo['c_gn_brent_90'], 
                             df_var_termo['c_gn_europa'], 
                             df_var_termo['c_gn_hh'], 
                             df_var_termo['c_gn_japao'],
                             df_var_termo['c_gn_renov'],
                             ], axis=1)

custos_historicos.to_csv(os.path.join(diretorio,'custos_historicos_v_gh.csv'), 
                         index=False, sep=';')


###############################################################################

# AVALIAÇÃO DE PORTFÓLIOS

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def alocacao_portfolio(n):
    n_aleatorios = np.random.rand(n)                                                    # combinação aleatória das fontes de geração
    return n_aleatorios / sum(n_aleatorios)

def desempenho_portfolio_ref(i, custo_fonte, COV):
    opcoes = custo_fonte.shape[1]
    m = [1 / (opcoes + 3)] * opcoes
    m[i] = 4 / (opcoes + 3)
    mix = np.asmatrix(m)
    med_ref = mix * custo_fonte.T                                                       # custo médio ponderado do portfolio
    var_ref = mix * COV * mix.T                                                         # variância de custo do portfolio
    print('Portfólio % 3d: ' % i, end='')
    for j in range(mix.shape[1]-1): 
        print('% 5.2f  ' % mix[0,j].item(), end='') 
    print('% 5.2f  ' % mix[0,-1].item()) 
    return med_ref, var_ref

def desempenho_portfolio(i, mix, COV):    
   # alocação de usina
    med = mix * custo_fonte.T                                                           # custo médio ponderado do portfolio
    dp = np.sqrt(mix * COV * mix.T)                                                     # desvio padrão do custo do portfolio
    print('Portfólio % 3d: ' % i, end='')
    for j in range(mix.shape[1]-1): 
        print('% 5.2f  ' % mix[0,j].item(), end='') 
    print('% 5.2f  ' % mix[0,-1].item()) 
    return med, dp


diretorio = os.getcwd()
fontes = ['uhe', 'eol', 'sol','carvao', 'carvao_renov', 'gn']
tipos_de_ute_gn = ['gn_0', 'gn_50', 'gn_90', 'gn_europa', 'gn_hh', 'gn_japao', 'gn_renov']
alternativas = fontes[0 : -1] + tipos_de_ute_gn
arquivo = 'custos_historicos_v_gh.csv'
custos_historicos = pd.read_csv(os.path.normcase(
        os.path.join(diretorio, arquivo)), engine='python', sep=';')
custos_historicos.dropna(inplace=True)
custos = custos_historicos.to_numpy()
custo_fonte = np.asmatrix(np.mean(custos, axis=0))          # custo médio esperado de cada fonte/tecnologia 
vec_custos = custos - custo_fonte
COV = np.asmatrix(np.cov(vec_custos.T))                     # matriz de covariâncias
n_portfolios = 1000                                         # portfolios gerados aleatoriamente
v_med, v_dp, v_mix = np.array([]), np.array([]), np.array([])

# ESTUDO: PORTFÓLIOS SELECIONADOS
mix_estudo = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],     # 1  única fonte 
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],     # 2
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],     # 3
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],     # 4
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],     # 5
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],     # 6
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],     # 7
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],     # 8
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],     # 9
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],     # 10
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],     # 11
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],     # 12
          
              [.5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 13  combinações com 50% hidro
              [.5, 0, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 14
              [.5, 0, 0, .5, 0, 0, 0, 0, 0, 0, 0, 0],   # 15
              [.5, 0, 0, 0, .5, 0, 0, 0, 0, 0, 0, 0],   # 16
              [.5, 0, 0, 0, 0, .5, 0, 0, 0, 0, 0, 0],   # 17
              [.5, 0, 0, 0, 0, 0, .5, 0, 0, 0, 0, 0],   # 18
              [.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .5],   # 19
          
              [.4, .6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 20  combinações hidro-eólico
              [.2, .8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 21
          
              [0, .5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 22  combinações com 50% eólico
              [0, .5, 0, .5, 0, 0, 0, 0, 0, 0, 0, 0],   # 23
              [0, .5, 0, 0, .5, 0, 0, 0, 0, 0, 0, 0],   # 24
              [0, .5, 0, 0, 0, .5, 0, 0, 0, 0, 0, 0],   # 25
              [0, .5, 0, 0, 0, 0, .5, 0, 0, 0, 0, 0],   # 26
              [0, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, .5],   # 27
         
              [0, .6, 0, 0, .4, 0, 0, 0, 0, 0, 0, 0],   # 28  combinações eólico-renovado
              [0, .8, 0, 0, .2, 0, 0, 0, 0, 0, 0, 0],   # 29
              [0, .6, 0, 0, 0, 0, 0, 0, 0, 0, 0, .4],   # 30
              [0, .8, 0, 0, 0, 0, 0, 0, 0, 0, 0, .2]]   # 31

for i in range(len(mix_estudo)):
    mix = np.asmatrix(mix_estudo[i])
    med, dp = desempenho_portfolio(i, mix, COV)
    v_med = np.append(v_med, med)
    v_dp = np.append(v_dp, dp)
    v_mix = np.append(v_mix, mix)
opcoes = mix.size
print()
print()
print('Matriz de Covariâncias')
print()
for i in range(opcoes):
    print('{} \t'.format(alternativas[i]),end='')
print()
for lin in range(0, opcoes):
    for col in range(0, opcoes):
        print('% 5.2f \t' % COV[lin, col].item(), end='')
    print()  
print()

# Plotar Resultados
fig = plt.figure(figsize=(16, 8))
plt.plot(v_dp, v_med, 'o', color='yellowgreen', markersize=4)
i = 0
for x, y in zip(v_dp, v_med):
    i += 1
    plt.text(x, y, str(i), color='gray', fontsize=9)
plt.xlabel('desvio padrão (R$/MWh)', fontsize=14)
plt.ylabel('média (R$/MWh)', fontsize=14)
plt.title('Custo total de suprimento de diferentes portfolios de expansão de geração', fontsize=16)
plt.show()

# PORTFÓLIOS ALEATÓRIOS
for i in range(n_portfolios):
    mix_de_fontes = np.array(alocacao_portfolio(len(fontes)))                                 # alocação entre fontes
    mix_de_ute_gn = np.array(alocacao_portfolio(len(tipos_de_ute_gn)) * mix_de_fontes[-1])    # alocacao de termeletricas
    mix = np.asmatrix(np.concatenate((mix_de_fontes[0 : -1], mix_de_ute_gn), axis = 0)) 
    med, dp = desempenho_portfolio(i, mix, COV)
    v_med = np.append(v_med, med)
    v_dp = np.append(v_dp, dp)
    v_mix = np.append(v_mix, mix)
    
# Plotar Resultados
fig = plt.figure(figsize=(16, 8))
plt.plot(v_dp, v_med, 'o', color='yellowgreen', markersize=4)
i = 0
# for x, y in zip(v_dp, v_med):
#     i += 1
#     plt.text(x, y, str(i), color='gray', fontsize=9)
plt.xlabel('desvio padrão (R$/MWh)', fontsize=14)
plt.ylabel('média (R$/MWh)', fontsize=14)
plt.title('Custo total de suprimento de diferentes portfolios de expansão de geração', fontsize=16)
plt.show()

# # para visualizar mix dos resultados 
# # n = ponto desejado
# n = 1799
# column1 = alternativas
# column2 = []
# for i in range((n-1)*12,(n-1)*12+12):
#     column2.append('{:.2f}'.format(v_mix[i]))
# for c1, c2 in zip(column1, column2):
#     print ('%-16s %s'% (c1, c2))
    
    