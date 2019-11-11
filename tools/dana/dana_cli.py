# Copyright 2018 The MACE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fcntl
import json
import os

try:
    from urllib2 import Request as SixUrlRequest
    from urllib2 import urlopen as SixUrlOpen
except ImportError:
    from urllib.request import Request as SixUrlRequest
    from urllib.request import urlopen as SixUrlOpen

DANA_HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Content-Type': 'application/json',
}


class DanaTrend:
    SMALLER = 'smaller'
    HIGHER = 'higher'


class DanaCli:
    def __init__(self, domain, serie_cfg_path):
        self._domain = domain
        self._serie_cfg_path = serie_cfg_path
        self._exist_series = self.load_series_from_cfg()

    def service_available(self):
        index_url = "http://%s" % self._domain
        request = SixUrlRequest(index_url)
        response = None
        try:
            response = SixUrlOpen(request).read()
        except Exception:
            print("Dana service is not available.")
        return (response is not None)

    def serie_available(self, serie_id):
        return (serie_id in self._exist_series)

    # ref: https://github.com/google/dana/blob/master/docs/Apis.md#addbuild
    def add_build(self, project_id, build_id, build_hash, abbrev_hash,
                  author_name, author_mail, subject, url=None, override=False):
        build_data = {
            'projectId': project_id,
            'build': {
                'buildId': build_id,
                'infos': {
                    'hash': build_hash,
                    'abbrevHash': abbrev_hash,
                    'authorName': author_name,
                    'authorEmail': author_mail,
                    'subject': subject,
                    'url': url
                }
            },
            'override': override
        }
        request_url = "http://%s/apis/addBuild" % self._domain
        succ = self.post_data(request_url, build_data)
        if succ:
            print("Add a build id, build_data: %s" % build_data)
        else:
            print("Add build id failed. build_data: %s" % build_data)
        return succ

    # ref: https://github.com/google/dana/blob/master/docs/Apis.md#addserie
    def add_benchmark_serie(self, project_id, serie_id, range, required,
                            trend, base_id=None, description=None, infos=None,
                            override=True):
        serie_data = {
            'projectId': project_id,
            'serieId': serie_id,
            'analyse': {
                'benchmark': {
                    'range': range,
                    'required': required,
                    'trend': trend
                }
            },
            'override': override
        }
        succ = self.add_serie(serie_data, base_id, description, infos)
        if succ:
            print("Add a benchmark serie, serie_data: %s" % serie_data)
        else:
            print("Add benchmark serie failed. serie_data: %s" % serie_data)
        return succ

    # ref: https://github.com/google/dana/blob/master/docs/Apis.md#addserie
    def add_test_serie(self, project_id, serie_id, base_id=None,
                       propagate=True, description=None, infos=None,
                       override=True):
        serie_data = {
            'projectId': project_id,
            'serieId': serie_id,
            'analyse': {
                'test': {
                    'propagate': propagate
                }
            },
            'override': override
        }
        succ = self.add_serie(serie_data, base_id, description, infos)
        if succ:
            print("Add a test serie, serie_data: %s" % serie_data)
        else:
            print("Add test serie failed. serie_data: %s" % serie_data)
        return succ

    # ref: https://github.com/google/dana/blob/master/docs/Apis.md#addsample
    # for test, value is true/false
    # for benchmark, value is zero integer
    def add_sample(self, project_id, serie_id, build_id,
                   value, override=False, skip_analysis=False):
        samples = [{
            'buildId': build_id,
            'value': value
        }]
        return self.add_samples(project_id, serie_id,
                                samples, override, skip_analysis)

    # ref: https://github.com/google/dana/blob/master/docs/Apis.md#addsample
    # samples should be an array includes {'buildId': build_id, 'value': value}
    def add_samples(self, project_id, serie_id, samples,
                    override=False, skip_analysis=False):
        samples_data = {
            'projectId': project_id,
            'serieId': serie_id,
            'samples': samples,
            'override': override,
            'skipAnalysis': skip_analysis
        }
        request_url = "http://%s/apis/addSample" % self._domain
        succ = self.post_data(request_url, samples_data)
        if succ:
            print("Add samples, samples_data: %s" % samples_data)
        else:
            print("Add samples failed. samples_data: %s" % samples_data)
        return succ

    def add_serie(self, serie_data, base_id=None,
                  description=None, infos=None):
        if base_id is not None:
            serie_data['analyse']['base'] = base_id
        if description is not None:
            serie_data['description'] = description
        if infos is not None:
            serie_data['infos'] = infos
        request_url = "http://%s/apis/addSerie" % self._domain
        succ = self.post_data(request_url, serie_data)
        if succ:
            succ = self.write_serie_to_cfg(serie_data['serieId'])
        return succ

    def post_data(self, request_url, json_data):
        json_data = json.dumps(json_data).replace('\'', '\"')
        data = json_data.encode()
        request = SixUrlRequest(request_url,
                                headers=DANA_HEADERS,
                                data=data)
        try:
            response = SixUrlOpen(request).read()
        except Exception as e:
            print("Http error, url=%s\npost_data=%s\n%s" %
                  (request_url, json_data, e))
            return False
        result = response.decode()
        if len(result) < 30 and "successfull" in result:
            return True
        else:
            return False

    def load_series_from_cfg(self):
        exist_series = []
        if os.path.exists(self._serie_cfg_path):
            with open(self._serie_cfg_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                for serie in f.readlines():
                    exist_series.append(serie.strip())
        return exist_series

    def write_serie_to_cfg(self, serie_id):
        if len(serie_id) == 0:
            print('serie_id is empty.')
            return False
        if serie_id in self._exist_series:
            return True

        with open(self._serie_cfg_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write("%s\n" % serie_id)
            self._exist_series.append(serie_id)
        return True
