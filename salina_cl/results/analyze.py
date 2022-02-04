#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import salina
import re
pd.options.mode.chained_assignment = None

def extract_scenario(log):
    values={}
    for k in log.hps:
        if k.startswith("scenario"):
            values[k]=log.hps[k]
    return values

def has_scenario(log,scenario):
    s=extract_scenario(log)
    s=str(s)
    return s==str(scenario)
    
def unique_scenarios(logs):
    _unique_scenarios={}
    for l in logs.logs:
        scenario=extract_scenario(l)
        _unique_scenarios[str(scenario)]=scenario
    _unique_scenarios=[v for s,v in _unique_scenarios.items()]
    return _unique_scenarios

def generate_scenario_html(scenario):
    results=["<h2>Scenario</h2>"]
    results.append("<ul>")
    
    for k,v in scenario.items():
        results.append("<li><b>"+k+"</b> ="+str(v)+"</li>")
    results.append("</ul>")
    return "".join(results)

def generate_hps_html(hps):
    results=["<h3>Hyper-parameters</h3>"]
    results.append("<ul>")
    
    for k,v in hps.items():
        if k.startswith("model"):
            results.append("<li><b>"+k+"</b> ="+str(v)+"</li>")
    results.append("</ul>")
    return "".join(results)

def generate_reward_table_html(reward_mean,reward_std):
    results=["<h3>Reward</h3>"]
    results.append("<table>")
    n,_=reward_mean.shape
    
    results.append("<tr><td>Task \\ Stage </td>")
    for stage in range(n): results.append("<td><b>"+str(stage)+"</b></td>")
    results.append("</tr>")
    
    for task in range(n):
        results.append("<tr><td><b>"+str(task)+"</b></td>")
        for stage in range(n): 
            r = str(round(reward_mean[task][stage],0))
            rs = str(round(reward_std[task][stage],0))
            if rs != 0:
                results.append("<td>"+r+" <small><i>±"+rs+"</i></small></td>")
            else:
                results.append("<td>"+r)
        results.append("</tr>")
    results.append("</table>")
    return "".join(results)

def generate_key_metrics_html(reward_mean,reward_std):
    results=["<h3>Key metrics</h3>"]
    results.append("<ul>")
    final_avg_perf = round(reward_mean[:,-1].mean(),0)
    results.append("<li><b>Final average perf</b> ="+str(final_avg_perf)+"</li>")
    forward_transfer = round(reward_mean.T[np.triu_indices(reward_mean.shape[0], k = 1)].mean(),0)
    results.append("<li><b>Forward transfer</b> ="+str(forward_transfer)+"</li>")
    backward_transfer = round((reward_mean[:,-1] - reward_mean.diagonal()).mean(),2)
    results.append("<li><b>Forgetting</b> ="+str(backward_transfer)+"</li>")
    results.append("</ul>")
    return "".join(results)


# Remove the run information and extrat the hps as a str in each log
def extract_hps(log):
    values={}
    for k,v in log.hps.items():
        if not "seed" in k and not k.endswith("device"):
            values[k]=v
    return values

def analyze_runs(logs):
    print("Analyzing ",len(logs)," logs")
    hps = extract_hps(logs[0])
    dfs = []
    for log in logs:
        df = log.to_dataframe()
        _cols = [c for c in df.columns if c.startswith("evaluation")]+["iteration"]  
        df = df[_cols]
        dfs.append(df)
    
    df=pd.concat(dfs)
    df_mean=df.groupby("iteration",as_index=False).mean()
    df_std=df.groupby("iteration",as_index=False).std()
    columns=[c for c in df_mean.columns if not c=="iteration"]
    df_mean=df_mean.dropna(subset=columns,how="all")
    df_std=df_std.dropna(subset=columns,how="all")
    n_tasks=df_mean["iteration"].max()+1
    #Collection reward
    r_mean=np.zeros((n_tasks,n_tasks))
    r_std=np.zeros((n_tasks,n_tasks))
    memory_mean=np.zeros((n_tasks,))
    memory_std=np.zeros((n_tasks,))
    for task in range(n_tasks):
        for stage in range(n_tasks):
            n="evaluation/"+str(task)+"/avg_reward"
            d=df_mean[df_mean["iteration"]==stage]
            
            reward_mean=d.iloc[0][n]
            memory_mean[stage]=d.iloc[0]["evaluation/memory/n_parameters"]
            r_mean[task][stage]=round(reward_mean,0)
            
            d=df_std[df_std["iteration"]==stage]
            try:
                reward_std=d.iloc[0][n]
                memory_std[stage]=d.iloc[0]["evaluation/memory/n_parameters"]
            except:
                reward_std = 0
                memory_std[stage]=memory_mean[stage]
            
            r_std[task][stage]=reward_std
    return r_mean,r_std,memory_mean,memory_std,hps

def analyze_scenario(logs,scenario):
    h=generate_scenario_html(scenario)
    display(HTML(h))
    per_hps={}
    for log in logs.logs:
        if not has_scenario(log,scenario):
            continue
        h=extract_hps(log)
        str_h=str(h)
        if not str_h in per_hps:
            per_hps[str_h]=[]
        per_hps[str_h].append(log)
    
    print("Found ",len(per_hps)," different Hps values")
    
    for i,h in enumerate(per_hps):
        reward_mean, reward_std, memory_mean, memory_std, hps = analyze_runs(per_hps[h])
        #Generate HTML
        display(HTML("<h2>#"+str(i+1)+"</h2>"))
        h = generate_key_metrics_html(reward_mean,reward_std)
        display(HTML(h))
        h = generate_reward_table_html(reward_mean,reward_std)
        display(HTML(h))
        h = generate_hps_html(hps)
        display(HTML(h))
        display(HTML("<h2>"+("_"*10)+"</h2>"))

def agregate_experiments(path):
    logs = salina.logger.read_directory(path,use_bz2=True)
    dfs = []
    d_id = {}
    d_hp= {}
    d_logs = {}
    i = 0
    scenarios = unique_scenarios(logs)
    for scenario in scenarios:
        for log in logs.logs:
            if has_scenario(log,scenario):
                df = log.to_dataframe()
                _cols = [c for c in df.columns if (c.startswith("evaluation/") or c.startswith("model/"))]+["iteration"]
                df = df[_cols]
                n_tasks = 1+max([int(re.findall("/([0-9]+)/",x)[0]) for x in df.columns if ("evaluation/" in x) and ("avg_reward" in x)])
                df = df[df["iteration"] < n_tasks]
                hp = extract_hps(log)
                hp_key = str({k:v for k,v in hp.items() if not "seed" in k})
                if not (hp_key in d_id):
                    d_id[hp_key] = i
                    d_hp[i] = hp
                    i += 1
                d_logs[d_id[hp_key]] = d_logs.get(d_id[hp_key],[]) + [log]
                df["id"] = d_id[hp_key]
                df["scenario"] = scenario["scenario/name"]
                dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs,d_logs,d_hp

def sort_best_experiments(df, top_k = 1):
    nb_tasks = max([int(re.findall("/([0-9]+)/",x)[0]) for x in df.columns if ("evaluation/" in x) and ("avg_reward" in x)])
    df = df[df["iteration"] == nb_tasks]
    df["evaluation/global_avg_reward"] = df[[c for c in df.columns if c.startswith("evaluation/")]].mean(axis=1)
    df = df[["id","evaluation/global_avg_reward"]].groupby("id").mean().sort_values(by="evaluation/global_avg_reward",ascending=False)
    best_ids = df.index[:top_k]
    return best_ids

def display_best_experiments(PATH,top_k=1):
    dfs,d_logs,d_hp = agregate_experiments(PATH)
    best_ids = sort_best_experiments(dfs,top_k = top_k)
    #analyze_scenario(best_logs,)

    #h = generate_scenario_html(scenario)
    #display(HTML(h))
    display(HTML("<h2>"+("_"*100)+"</h2>"))
    for i,best_id in enumerate(best_ids):
        reward_mean, reward_std, memory_mean, memory_std, hps = analyze_runs(d_logs[best_id])
        #Generate HTML
        display(HTML("<h2>#"+str(i+1)+"</h2>"))
        h = generate_key_metrics_html(reward_mean,reward_std)
        display(HTML(h))
        h = generate_reward_table_html(reward_mean,reward_std)
        display(HTML(h))
        h = generate_hps_html(hps)
        display(HTML(h))
        display(HTML("<h2>"+("_"*100)+"</h2>"))