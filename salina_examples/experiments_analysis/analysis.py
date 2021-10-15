#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
from tqdm import tqdm

import salina.logger

st.set_option("deprecation.showPyplotGlobalUse", False)


@st.cache(allow_output_mutation=True)
def read_data(directories):
    logs = salina.logger.read_directories(directories)
    dfs = []
    print("Converting logs to dataframes")
    for l in tqdm(logs.logs):
        dfs.append(l.to_dataframe(with_hps=True))
    return logs, dfs


def m_hps(logs):
    return


def render_plot(logs):
    m_hps = [k for k in logs.hps() if len(logs.unique_hps(k)) > 1]
    st.subheader("Plotting")
    st.text("Choose what to plot...")
    with st.beta_container():
        y_box = st.selectbox("Y: ", logs.columns())
        style_box = st.multiselect("Style: ", m_hps)
        cols_box = st.multiselect("Cols: ", m_hps)
        rows_box = st.multiselect("Rows: ", m_hps)
        use_smoothing = st.checkbox("Use smoothing")
        with st.beta_expander("Click to see smoothing parameters..."):
            every_n = st.slider(
                "Every n :",
                min_value=1,
                max_value=10000,
                value=1,
                help="Sub-sample every n iterations to speedup-drawing",
            )
            smoothing_size = st.slider(
                "Smoothing :",
                min_value=1,
                max_value=100,
                value=1,
                help="Choosing smoothing window size",
            )
    v = {"y": y_box, "style": style_box, "cols": cols_box, "rows": rows_box}
    if use_smoothing:
        v["smoothing"] = {"every_n": every_n, "smoothing_size": smoothing_size}
    return v


def plot(dataframes, render_plot):
    y = render_plot["y"]
    __df = []
    if "smoothing" in render_plot:
        st.text("Interpolating missing values and smoothing")
        for d in dataframes:
            d[y] = d[y].fillna(method="bfill")
            d = d[d["iteration"] % render_plot["smoothing"]["every_n"] == 0]
            d[y] = (
                d[y]
                .rolling(render_plot["smoothing"]["smoothing_size"], min_periods=1)
                .sum()
            )
            __df.append(d)
    else:
        for d in dataframes:
            d[y] = d[y].fillna(method="bfill")
            __df.append(d)
        __df = dataframes
    dataframe = pd.concat(__df)
    st.text("Kept: " + str(len(dataframe.index)) + " lines.")
    st.text("Drawing...")

    style = render_plot["style"]
    if len(style) == 0:
        style = None
    if style is not None:
        style = ["_hp/" + k for k in style]
        if len(style) == 1:
            style = style[0]
    rows = render_plot["rows"]
    if len(rows) == 0:
        rows = None
    if rows is not None:
        rows = ["_hp/" + k for k in rows]
        if len(rows) == 1:
            rows = rows[0]
    cols = render_plot["cols"]
    if len(cols) == 0:
        cols = None
    if cols is not None:
        cols = ["_hp/" + k for k in cols]
        if len(cols) == 1:
            cols = cols[0]

    salina.logger.plot_dataframe(
        dataframe, x="iteration", y=y, style=style, row=rows, col=cols
    )
    st.pyplot()


def render_filter(logs):
    st.subheader("Filtering")
    st.text("Choose which hyper-parameters values to keep for analysis.")
    filter_box = {}
    for k in logs.hps():
        u = logs.unique_hps(k)
        if len(u) > 1:
            with st.beta_container():
                filter_box[k] = st.multiselect(k, u, default=u)
    return filter_box


def filter_dataframes(logs, dataframes, filter_box):
    st.text("Filtering lines...")
    filter_dict = {}
    for k in logs.hps():
        filter_dict[k] = []
        u = logs.unique_hps(k)
        if len(u) > 1:
            for uu in u:
                if not uu in filter_box[k]:
                    filter_dict[k].append(uu)

    _dataframes = {k: d for k, d in enumerate(dataframes)}
    for k in filter_dict:
        for u in filter_dict[k]:
            for _k, _l in _dataframes.items():
                if logs.logs[_k].hps[k] == u:
                    _dataframes[_k] = None
    _dataframes = [d for _, d in _dataframes.items() if not d is None]
    st.text("... Kept " + str(len(_dataframes)) + " logs..")
    return _dataframes


def render_fixed_values(logs):
    st.subheader("Fixed Hyper-parameters")
    with st.beta_expander("Click to see values..."):
        d = {}
        for k in logs.hps():
            u = logs.unique_hps(k)
            if len(u) == 1:
                d[k] = u[0]
        st.json(d)


def render_curve(logs, dataframes):
    st.header("Drawing performance...")
    cols = st.beta_columns([1, 2])
    with st.form("my_form"):
        submitted = st.form_submit_button("Submit")

        with cols[0]:
            log = logs.logs[0]
            for k, v in enumerate(log.values["evaluation/reward"]):
                if not v is None:
                    print(k, v)

            filter_box = render_filter(logs)
            _render_plot = render_plot(logs)

        with cols[1]:
            if submitted:
                dataframes = [copy.deepcopy(d) for d in dataframes]
                _dataframes = filter_dataframes(logs, dataframes, filter_box)
                plot(_dataframes, _render_plot)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    directory = sys.argv[1]
    st.set_page_config(layout="wide")
    st.title("Analysis of '" + directory + "'")
    st.header("Logs")
    directories = salina.logger.get_directories(directory)
    data_load_state = st.text("...Loading data...(can take a few minutes)")
    logs, dataframes = read_data(directories)
    data_load_state.text("...Loading data...done!")
    st.write("**Number of logs: **", len(logs))
    render_fixed_values(logs)
    render_curve(logs, dataframes)
