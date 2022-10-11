import matplotlib.pyplot as plt
import os,pickle,tqdm,json
import warnings,json,itertools
from collections import Counter
import datetime as dt
import networkx as nx
import pandas as pd
import numpy as np
from multiprocessing import Pool,cpu_count
warnings.simplefilter('ignore')

def convert_key_tag_top(df):
    
    cols_to_convert = ['keywords','tags','topics']
    
    keywords,tags,topics = [],[],[]
    for i,v in tqdm.tqdm(df.iterrows()):
        if v['keywords']==v['keywords']:
            keywords.append((i,list(json.loads(
                v['keywords']).keys())))
        else:keywords.append((i,[]))
        if v['tags']==v['tags']:
            tags.append((i,list(json.loads(v['tags']))))
        else:tags.append((i,[]))        
        if v['topics']==v['topics']:
            topics.append((i,list(json.loads(v['topics']))))
        else:topics.append((i,[]))

    to_join = []
    for ls_nm,ls_ls in zip(cols_to_convert,
                           [keywords,tags,topics]):
        to_join.append(pd.DataFrame(ls_ls).rename(
            columns={1:ls_nm}).set_index(0))

    df = df.drop(cols_to_convert,axis=1).join(
        pd.concat(to_join,axis=1))
    
    for col in cols_to_convert:
        df[f'n_{col}'] = df.apply(
            (lambda x:len(x[col])),axis=1)
        df[f'1_{col}'] = df.apply(
            (lambda x:int(len(x[col])<=1)),axis=1)
        
    return df

def convert_key_tag_top_novelty(df,date_col_name='post_date'):
    
    dfs = []
    to_remake = ['keywords','topics','tags']
    for col in to_remake:
        ls = []
        for i,v in tqdm.tqdm(df.iterrows()):
            if v[col]==v[col]:
                ls.append((i,sorted(json.loads(
                    v[col].replace("\'",'\"')))))
            else:ls.append((i,[]))
        df_ = pd.DataFrame(ls).rename(columns={1:col}).set_index(0)
        dfs.append(df_)

    dfs = pd.concat(dfs,axis=1)
    df = df.drop(to_remake,axis=1).join(dfs)
    
    return df.rename(columns = {date_col_name:'datetime'})

def flatten_list_to_set(ls_of_ls):
    return set([it for sl in ls_of_ls for it in sl])

def plot_weighted_circular(G,weighted=True,
                           plot=True,community=None):
    
    if plot==True:
        plt.figure(figsize=(5,5))
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G,pos,node_size=20)
        
        if weighted:
            
            width = np.array(
                [info[2]['weight'] for\
                 info in G.edges(data=True)])
            
            if max(width)>10:
                width=width/10.0
                
            if community:
                
                edge_list = [
                    i for i,v in same_diff_comm.items() if v==True]
                nx.draw_networkx_edges(
                    G,pos,edge_list,width=width,alpha=0.25)
                
                edge_list = [
                    i for i,v in same_diff_comm.items() if v==False]
                nx.draw_networkx_edges(
                    G,pos,edge_list,edge_color='r',
                    width=width,alpha=0.25)  
                
            else:
                nx.draw_networkx_edges(G,pos,width=width,alpha=0.5)
                
        else:
            
            edge_list = [
                i for i,v in same_diff_comm.items() if v==True]
            nx.draw_networkx_edges(
                G,pos,edge_list,alpha=0.25)
            
            edge_list = [
                i for i,v in same_diff_comm.items() if v==False]
            nx.draw_networkx_edges(
                G,pos,edge_list,
                edge_color='r',alpha=0.25) 
            
        plt.show()
        
def get_pairwise_combo(ls,self_loop=True):
    
    if len(ls)>=1:
        combo = list(itertools.combinations(ls,2))
        combo = list(tuple(sorted(i)) for i in combo)
        
        if self_loop:
            for it in ls:
                combo.append((it,it))
            
    else:
        combo = []
    
    return combo
        
def get_observed_freq(df,col='topics',self_loop=False):
    
    nodes = []
    for sublist in df[col].values:
        for item in sublist:
            nodes.append(item)
    all_edges = get_pairwise_combo(nodes,self_loop)
    
    obs_edges = []
    for i,v in df.iterrows():
        
        if len(v[col])==1:
            combo = get_pairwise_combo(v[col],self_loop)
        else:
            combo = get_pairwise_combo(v[col],False)
            
        if combo:
            for pair in combo:
                pair = tuple(sorted(pair))
                obs_edges.append(pair)
    obs_edges = dict(Counter(obs_edges))

    final_counts = {edge:0 for edge in all_edges}
    for edge,weight in obs_edges.items():
        final_counts[edge] = weight
        
    return final_counts

def get_expected_freq(
    observed_freq,plot=True,self_loop=False):
    
    G = nx.Graph()
    for edge,weight in observed_freq.items():
        if weight>0:
            G.add_edge(edge[0],edge[1],weight=weight)

    if plot:
        print('number of edges:',len(G.edges(data=True)))
        plot_weighted_circular(G) 
        
    try:
        swaps = nx.connected_double_edge_swap(G,10)
    except:
        swaps = nx.double_edge_swap(G)
        
    if plot: 
        print('number of swaps:',swaps)
        print('randomly assign edge weights')
        
    weights = [w for w in list(
        observed_freq.values()) if w>0]
    
    for edge in G.edges(data=True):
        chosen = np.random.choice(weights)
        weights.remove(chosen)
        G[edge[0]][edge[1]]['weight'] = chosen
    
    if plot:
        print('number of edges:',len(G.edges(data=True)))
        plot_weighted_circular(G)

    expected_freq = {tuple(sorted(i)):0 for\
                           i,_ in observed_freq.items()}
    for u,v,w in G.edges(data=True):
        expected_freq[tuple(sorted((u,v)))]=w['weight']
        
    return expected_freq

def uzzi2013(posts,measure_col='topics',interval='m',datetime_col='datetime'):

    # first convert the datetime to the time interval we want
    # then we can look at the subgraph for each time interval
    try:
        posts[datetime_col] = pd.to_datetime(
            posts[datetime_col]).dt.to_period(interval)
    except: pass
    months = sorted(posts[datetime_col].value_counts().keys())
    n_posts_too_few_in_period = 0
    n_no_z_scores = 0

    # we look at subgraphs by periods
    # here we use month but we can use days
    # depending on the dataset
    for count,month in enumerate(months):
        
        period = posts[posts[datetime_col]==month]
        observed_freq = get_observed_freq(
            period,measure_col,True)
        all_topics = set(
            [it for sl in period[measure_col] for it in sl])

        # if during the time period there is less than 2 post
        # of if during the time period there are less than 5 topics
        # which means that there are too few nodes
        if (len(all_topics)>=4) & (len(period)>=2):

            for idx,val in tqdm.tqdm(
                period.iterrows(),
                desc=f'{count+1}/{len(months)}'):

                z_score = {}
                z_score['observed'] = {}
                combos = get_pairwise_combo(val[measure_col])

                for combo in combos:
                    z_score['observed'][combo] = observed_freq[combo]

                for sim in range(20):
                    expected_freq = get_expected_freq(
                        observed_freq,False,True)

                    z_score[f'expected_{sim}'] = {}
                    for combo in get_pairwise_combo(val[measure_col]):                                            
                        z_score[f'expected_{sim}'][combo] = \
                        expected_freq[combo]

                z_score = pd.DataFrame.from_dict(z_score)   

                obs = z_score['observed']
                exp = z_score.drop('observed',axis=1).mean(axis=1)
                std = z_score.drop('observed',axis=1).std(axis=1)
                z = ((obs-exp)/(std)).values
                z = [val for val in z if val==val]

                if z:
                    posts.loc[idx,'novelty_median'] = np.nanmedian(z)
                    posts.loc[idx,'novelty_tenth'] = np.nanpercentile(z,10)
                else:
                    n_no_z_scores+=1
        else:
            n_posts_too_few_in_period+=1
            
    return posts

def foster2015(posts,measure_col='topics',
               interval='m',detection_method=None,
               datetime_col='datetime'):
    try:
        posts[datetime_col] = pd.to_datetime(
            posts[datetime_col]).dt.to_period(interval)
    except:pass
    months = sorted(posts[datetime_col].value_counts().keys())
    n_posts_too_few_in_period = 0
    n_no_z_scores = 0

    # we look at subgraphs by periods
    # here we use month but we can use days
    # depending on the dataset
    for count,month in enumerate(months):
        
        period = posts[posts[datetime_col]==month]
        observed_freq = get_observed_freq(
            period,measure_col,True)
        all_topics = set(
            [it for sl in period[measure_col] for it in sl])
        G = nx.Graph()

        for (u,v),w in observed_freq.items():
            G.add_edge(u,v,weight=w)

        if detection_method=='louvian':
            communities = nx.algorithms.community.louvain_communities(
                G,weight='weight')
        else:
            communities = nx.algorithms.community.greedy_modularity_communities(
                G,weight='weight')
        communities = {it:n for n,ls in enumerate(communities) for it in ls}
        same_diff_comm = {tuple(sorted((u,v))):\
                          communities[u]==communities[v] for u,v in G.edges()}

        if (len(all_topics)>=4) & (len(period)>=2):

            for idx,val in tqdm.tqdm(
                period.iterrows(),
                desc=f'{count+1}/{len(months)}'):
                combos = get_pairwise_combo(val[measure_col])
                counts = len(combos)
                # note that True means same community False means diff community
                scores = [0 if same_diff_comm[combo] else 1 for combo in combos]
                novelty = np.sum(scores)/counts
                posts.loc[idx,'novelty_foster'] = novelty
                
    return posts

def lee2015(posts,measure_col='topics',
            interval='m',
            datetime_col='datetime'):
    
    try:
        posts[datetime_col] = pd.to_datetime(
            posts[datetime_col]).dt.to_period(interval)
    except:pass
    
    months = sorted(posts[datetime_col].value_counts().keys())
    n_posts_too_few_in_period = 0
    n_no_z_scores = 0

    # we look at subgraphs by periods
    # here we use month but we can use days
    # depending on the dataset
    for count,month in enumerate(months):
        
        period = posts[posts[datetime_col]==month]
        observed_freq = get_observed_freq(
            period,measure_col,True); print('get observed freq...')
        all_topics = set(
            [it for sl in period[measure_col] for it in sl])
        Nt = len(observed_freq); print('get Nt...')
        Nit = len(all_topics)+1; print('get Nit...')
        Njt = len(all_topics)+1; print('get Njt...')
        denominator = Nit*Njt; print('get denominator...\n')

        for idx,val in tqdm.tqdm(period.iterrows(),
                                 desc=f'{count+1}/{len(months)}'):

            combos = get_pairwise_combo(val[measure_col])
            commonness = []
            for combo in combos:
                numerator = observed_freq[combo]*Nt
                commonness.append(numerator/denominator)  

            commonness = [c for c in commonness if c>0]
            commonness = np.percentile(commonness,10)
            posts.loc[idx,'commonness_lee'] = commonness

    return posts

def uzzi2013_month(item):
    
    count,period,measure_col = item
    observed_freq = get_observed_freq(
        period,measure_col,True)
    all_topics = set(
        [it for sl in period[measure_col] for it in sl])

    # if during the time period there is less than 2 post
    # of if during the time period there are less than 5 topics
    # which means that there are too few nodes
    
    if (len(all_topics)>=4) & (len(period)>=2):

        for idx,val in tqdm.tqdm(period.iterrows()):

            z_score = {}
            z_score['observed'] = {}
            combos = get_pairwise_combo(val[measure_col])

            for combo in combos:
                z_score['observed'][combo] = observed_freq[combo]

            for sim in range(20):
                expected_freq = get_expected_freq(
                    observed_freq,False,True)

                z_score[f'expected_{sim}'] = {}
                for combo in get_pairwise_combo(val[measure_col]):                                            
                    z_score[f'expected_{sim}'][combo] = \
                    expected_freq[combo]

            z_score = pd.DataFrame.from_dict(z_score)   
            obs = z_score['observed']
            exp = z_score.drop('observed',axis=1).mean(axis=1)
            std = z_score.drop('observed',axis=1).std(axis=1)
            z = ((obs-exp)/(std)).values
            z = [val for val in z if val==val]

            if z:
                period.loc[idx,'novelty_median'] = np.nanmedian(z)
                period.loc[idx,'novelty_tenth'] = np.nanpercentile(z,10)
            else:
                period.loc[idx,'novelty_median'] = np.NaN
                period.loc[idx,'novelty_tenth'] = np.NaN
    
    else:
        period.loc[:,'novelty_median'] = np.NaN
        period.loc[:,'novelty_tenth'] = np.NaN
        
    return period

def uzzi2013_parallel(
    posts,measure_col='topics',interval='m',
    datetime_col='datetime'):
    
    # first convert the datetime to the time interval we want
    # then we can look at the subgraph for each time interval
    try:
        posts[datetime_col] = pd.to_datetime(
            posts[datetime_col]).dt.to_period(interval)
    except: pass

    months = sorted(posts[datetime_col].value_counts().keys())
    n_posts_too_few_in_period = 0
    n_no_z_scores = 0
    iteration = [(
        count,posts[posts[datetime_col]==month],
        measure_col) for count,month in enumerate(months)]

    print(f'no. of cores found: {cpu_count()}\n')
    n_processes = min(len(iteration),cpu_count())
    print(f'relying on {n_processes} processes')
    pool = Pool(processes=n_processes)
    results = list(tqdm.tqdm(pool.map(uzzi2013_month,iteration,100)))
    pool.close()
    pool.join()
    
    return pd.concat(results)

def get_novelty(df,measure_col='topics',interval='m',parallel=True):

    start_time = dt.datetime.now()
    
    if parallel:
        df = uzzi2013_parallel(
            df,
            measure_col=measure_col,
            interval=interval)
    else:
        df = uzzi2013(
            df,
            measure_col=measure_col,
            interval=interval)
        
    print(f"total processing time:",
          dt.datetime.now() - start_time)
    return df


posts = convert_key_tag_top_novelty(pd.read_csv('./data/xhs/users_clean.csv',
                                                index_col='index'),'crawl_date')
_ = get_novelty(posts,parallel=True)
_ = get_novelty(posts,parallel=False)