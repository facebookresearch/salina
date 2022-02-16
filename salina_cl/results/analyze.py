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
import pickle
import os
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

def generate_reward_table_html(rewards,normalizing = False):
    reward_mean = rewards.mean(0)
    reward_std = rewards.std(0)
    results = ["<h3>Normalized rewards</h3>"] if normalizing else ["<h3>Rewards</h3>"]
    results.append("<table>")
    n,_=reward_mean.shape
    
    results.append("<tr><td>Task \\ Stage </td>")
    for stage in range(n): results.append("<td><b>"+str(stage)+"</b></td>")
    results.append("</tr>")
        
    for task in range(n):
        results.append("<tr><td><b>"+str(task)+"</b></td>")
        for stage in range(n): 
            r = stylify(reward_mean[task][stage],normalizing)
            rs = str(round(reward_std[task][stage],normalizing))
            if rs != 0:
                results.append("<td>"+r+" <small><i>± "+rs+"</i></small></td>")
            else:
                results.append("<td>"+r)
        results.append("</tr>")
    results.append("</table>")
    return "".join(results)

def stylify(r,normalizing = False):
    if normalizing:
        r = "<span style=\"color:#FF0000\">"+str(round(r,2))+"</span>" if r <1. else "<b style=\"color:#006400\">"+str(round(r,2))+"</b>"
    else:
        r = str(round(r,0))
    return r

def generate_memory_table_html(memory,normalizing = False):
    
    n = memory.shape[-1]
    if n > 1:
        memory_mean = memory.mean(0)
        memory_std = memory.std(0)
    else:
        memory_mean = memory[0]
        memory_std = np.zeros(memory.shape[1])
    

    results=["<h3>Memory</h3>"]
    results.append("<table>")
    results.append("<tr><td> Stage </td>")
    for stage in range(n): 
        results.append("<td><b>"+str(stage)+"</b></td>")
    results.append("</tr>")
    
    results.append("<tr><td> Nb_params </td>")
    for stage in range(n):
        if normalizing:
            r = str(round(memory_mean[stage],2))
            rs = str(round(memory_std[stage],2))
        else:
            r = str(int(memory_mean[stage]))
            rs = str(int(memory_std[stage]))
        if rs != 0:
            results.append("<td>"+r+" <small><i>± "+rs+"</i></small></td>")
        else:
            results.append("<td>"+r)
    results.append("</tr>")
    results.append("</table>")
    return "".join(results)

def generate_key_metrics_html(rewards,normalizing = False):
    
    results=["<h3>Key metrics</h3>"]
    results.append("<ul>")

    final_avg_perf_mean = round(rewards[:,:,-1].mean(),0)
    final_avg_perf_std = round(rewards[:,:,-1].mean(-1).std(),0)
    results.append("<li><b>Final average perf</b> ="+str(final_avg_perf_mean)+" <small><i>± "+str(final_avg_perf_std)+"</i></small></li>")

    forward = np.zeros((rewards.shape[0],))
    for i,r in enumerate(rewards):
        forward[i] = round(r.T[np.triu_indices(r.shape[0], k = 1)].mean(),0)
    forward_mean = round(forward.mean(),2)
    forward_std = round(forward.std(),2)
    results.append("<li><b>Forward Transfer</b> ="+str(forward_mean)+" <small><i>± "+str(forward_std)+"</i></small></li>")
    backward = np.zeros((rewards.shape[0],))
    for i,r in enumerate(rewards):
        backward[i] = round((r[:,-1] - r.diagonal()).mean(),2)
    backward_mean = round(backward.mean(),2)
    backward_std = round(backward.std(),2)
    results.append("<li><b>Forgetting</b> ="+str(backward_mean)+" <small><i>± "+str(backward_std)+"</i></small></li>")
    results.append("</ul>")
    return "".join(results)


# Remove the run information and extrat the hps as a str in each log
def extract_hps(log):
    values={}
    for k,v in log.hps.items():
        if not "seed" in k and not k.endswith("device"):
            values[k]=v
    return values

def extract_metrics(logs):
    print("Analyzing ",len(logs)," logs")
    hps = extract_hps(logs[0])
    dfs = []
    for i,log in enumerate(logs):
        df = log.to_dataframe()
        _cols = [c for c in df.columns if c.startswith("evaluation")]+["iteration"]  
        df = df[_cols]
        df["seed"] = i
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.dropna(subset=[c for c in df.columns if not ((c=="iteration") or (c=="seed"))],how="all")
    n_tasks = df ["iteration"].max() + 1
    n_seeds = df["seed"].max() + 1
    rewards = np.zeros((n_seeds,n_tasks,n_tasks))
    memory = np.zeros((n_seeds,n_tasks))
    for seed in range(n_seeds):
        for task in range(n_tasks):
            for stage in range(n_tasks):
                r_name ="evaluation/"+str(task)+"/avg_reward"
                
                d = df[(df["iteration"] == stage) & (df["seed"] == seed)]
                try:
                    memory[seed,stage] = d["evaluation/memory/n_parameters"]
                except:
                    memory[seed,stage] = 0
                try:
                    rewards[seed,task,stage] = round(d[r_name],0)
                except:
                    pass
    return rewards,memory,hps


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
    d_logs = {}
    i = 0
    problems = 0
    scenarios = unique_scenarios(logs)
    for scenario in scenarios:
        for log in logs.logs:
            if has_scenario(log,scenario):
                try:
                    df = log.to_dataframe()
                    _cols = [c for c in df.columns if (c.startswith("evaluation/") or c.startswith("model/"))]+["iteration"]
                    df = df[_cols]
                    n_tasks = 1+max([int(re.findall("/([0-9]+)/",x)[0]) for x in df.columns if ("evaluation/" in x) and ("avg_reward" in x)])
                    df = df[df["iteration"] < n_tasks]
                    hp = extract_hps(log)
                    hp_key = str({k:v for k,v in hp.items() if not "seed" in k})
                    if not (hp_key in d_id):
                        d_id[hp_key] = i
                        i += 1
                    d_logs[d_id[hp_key]] = d_logs.get(d_id[hp_key],[]) + [log]
                    df["id"] = d_id[hp_key]
                    df["scenario"] = scenario["scenario/name"]
                    dfs.append(df)
                except:
                    problems += 1
    print("Data loaded. Unable to load",problems,"experiment files.")
    dfs = pd.concat(dfs)
    return dfs,d_logs

def sort_best_experiments(df):
    nb_tasks = max([int(re.findall("/([0-9]+)/",x)[0]) for x in df.columns if ("evaluation/" in x) and ("avg_reward" in x)])
    df = df[df["iteration"] == nb_tasks]
    df["evaluation/global_avg_reward"] = df[[c for c in df.columns if c.startswith("evaluation/")]].mean(axis=1)
    df = df[["id","evaluation/global_avg_reward"]].groupby("id").mean().sort_values(by="evaluation/global_avg_reward",ascending=False)
    return df.index

def display_best_experiments(PATH,top_k=1, normalize_data = None, return_logs = False, force_loading = False):
    if os.path.exists(PATH+"experiments.dat") and (not force_loading):
        print("Experiments already agregated. Loading data...")
        with open(PATH+"experiments.dat", "rb") as f:
            data = pickle.load(f)
            dfs = data["dfs"]
            d_logs = data["d_logs"]
            best_ids = data["best_ids"]
    else:
        print("Agregating experiments...")
        dfs,d_logs = agregate_experiments(PATH)
        best_ids = sort_best_experiments(dfs)
        data = {"dfs":dfs,
                "d_logs":d_logs,
                "best_ids":best_ids}
        with open(PATH+"experiments.dat", "wb") as f:
            pickle.dump(data, f)

    normalizing = not (normalize_data is None)
    display(HTML("<h2>"+("_"*100)+"</h2>"))
    for i,best_id in enumerate(best_ids[:top_k]):
        rewards, memory,hps = extract_metrics(d_logs[best_id])

        #Normalizing data if possible
        if normalizing:
            with open(normalize_data, "rb") as f:
                d = pickle.load(f)
            n_seeds = rewards.shape[0]
            random_rewards = np.stack([d["random_rewards"] for _ in range(n_seeds)])
            baseline_rewards = np.stack([d["baseline_rewards"] for _ in range(n_seeds)])
            normalized_rewards = (rewards - random_rewards) / (baseline_rewards - random_rewards)
        #Generate HTML
        display(HTML("<h2>#"+str(i+1)+"</h2>"))
        h = generate_key_metrics_html(rewards)
        display(HTML(h))
        h = generate_reward_table_html(rewards)
        display(HTML(h))
        h = generate_reward_table_html(normalized_rewards,normalizing)
        display(HTML(h))
        h = generate_memory_table_html(memory,False)
        display(HTML(h))
        h = generate_hps_html(hps)
        display(HTML(h))
        display(HTML("<h2>"+("_"*100)+"</h2>"))

        with open(PATH+"experiments.dat", "wb") as f:
            pickle.dump(data, f)
    if return_logs:
        return dfs,d_logs, best_ids
