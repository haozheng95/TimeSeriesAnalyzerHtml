#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: yinhaozheng
@software: PyCharm
@file: test.py
@time: 2019-10-31 16:13
"""
import json

from phm.phm.demo import phm_16, conver_problem_desc, conver_features_rel, phm_17, conver_feature_desc, \
    conver_features_target_rel

__mtime__ = '2019-10-31'

if __name__ == '__main__':
    data = phm_16()
    feature_desc = data["feature_desc"]
    problem_desc = data["problem_desc"]
    features_rel = data["features_rel"]
    missing_data = data["missing_data"]
    problem_category = data["problem_category"]
    features_target_rel = data["features_target_rel"]
    # print(features_target_rel)
    v = conver_features_target_rel(features_target_rel, 2)
    print(v)
    # for k, v in conver_feature_desc(feature_desc["f_101"]).items():
    #     print(k, "--------->>>", v)
    # for k, v in conver_problem_desc(problem_desc).items():
    #     print(k, "--------->>>", v)
    # features_rel = json.loads(features_rel)
    # matrix = conver_features_rel(features_rel)
    # for k, v in features_rel["RETAINER_RING_PRESSURE"].items():
    #     print(k, "--------->>>", v)
    # print(missing_data)
    # print(problem_desc.replace('"',"").split(":"))
    # print(problem_category)
    # data2 = phm_16()
    # problem_desc = data2["problem_desc"]
    # print(problem_desc)
