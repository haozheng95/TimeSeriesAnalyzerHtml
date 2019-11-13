#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yinhaozheng
@software: PyCharm
@file: views.py
@time: 2019-10-25 09:38
"""
import hashlib
import json
import logging
import os
import time

from phm.phm.demo import phm_16, conver_problem_desc, conver_feature_desc, conver_features_rel, phm_17, \
    conver_features_target_rel

__mtime__ = '2019-10-25'
logger = logging.getLogger("django")
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

form_file_name = "exampleInputFile"
my_path = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(os.path.dirname(my_path), "static")
save_path = os.path.join(static_path, "taskFiles")


def index(request):
    if request.method == "GET":
        return render(request, "phm_index.html")

    task_id = upload_file(request)
    return HttpResponseRedirect("/phm-show/" + task_id)


def show(request, task_id):
    file = os.path.join(save_path, task_id)

    data = phm_16()
    problem_category = int(data["problem_category"])
    problem_desc = data["problem_desc"]
    feature_list = data["feature_list"]
    part_1 = None
    if problem_category == 1:
        part_1 = [problem_desc.replace('"', "").split(":")]
    else:
        part_1 = conver_problem_desc(problem_desc)

    part_2_list = data["feature_desc"]
    part_2_view_list = []
    for k, v in part_2_list.items():
        temp = dict(
            feature=k,
            feature_desc=conver_feature_desc(v)
        )
        part_2_view_list.append(temp)
        break

    features_rel = data["features_rel"]
    features_target_rel = conver_features_target_rel(data["features_target_rel"], problem_category)
    features_rel = json.loads(features_rel)
    matrix = conver_features_rel(features_rel)

    if problem_category == 1:
        return render(request, "phm_show_1.html",
                      dict(
                          part_1=part_1,
                          part_2_view_list=part_2_view_list,
                          matrix=matrix,
                          problem_category=problem_category,
                          feature_list=feature_list,
                          features_target_rel=features_target_rel,
                      ))

    return render(request, "phm_show_1.html",
                  dict(histogram_list=part_1["histogram_list"],
                       part_1_descriptive_statistics_summary=part_1["descriptive_statistics_summary"],
                       part_1_skewness=part_1["skewness"],
                       part_1_kurtosis=part_1["kurtosis"],
                       part_2_view_list=part_2_view_list,
                       matrix=matrix,
                       problem_category=problem_category,
                       feature_list=feature_list,
                       features_target_rel=features_target_rel
                       ))


def upload_file(request):
    file_name = str(time.time()) + "__" + request.FILES[form_file_name].name
    hl = hashlib.md5()
    hl.update(file_name.encode(encoding='utf-8'))
    file_name = hl.hexdigest()
    task_path = os.path.join(save_path, file_name)
    handle_uploaded_file(request.FILES[form_file_name].file, task_path)
    return file_name


def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f:
            destination.write(chunk)
