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

from dana.dana_cli import DanaCli
from dana.dana_cli import DanaTrend
import os
import re
import sh
import time

DANA_DOMAIN = 'localhost:7000'
DEFAULT_RANGE = '5%'
DEFAULT_REQUIRED = 2
MACE_PROJECT_ID = 'Mace'
SERIE_CFG_PATH = "%s/.dana/mace_dana.conf" % os.environ['HOME']
SERVICE_LIFE = 60  # 60s


class DanaUtil:
    def __init__(self, domain=DANA_DOMAIN, serie_cfg_path=SERIE_CFG_PATH):
        dir_path = os.path.dirname(serie_cfg_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        self._dana_cli = DanaCli(domain, serie_cfg_path)
        self._valid_time = None
        self._build_id = None
        self._service_available = False

    def fast_report_test(self, table_name, base_name, device_name,
                         soc, target_abi, runtime, value):
        return self.fast_report_sample(True, table_name, base_name,
                                       device_name, soc, target_abi,
                                       runtime, value)

    def fast_report_benchmark(self, table_name, base_name, device_name,
                              soc, target_abi, runtime, value, trend):
        return self.fast_report_sample(False, table_name, base_name,
                                       device_name, soc, target_abi,
                                       runtime, value, trend)

    def fast_report_sample(self, is_test, table_name, base_name, device_name,
                           soc, target_abi, runtime, value, trend):
        if not self.service_available():
            return True
        serie_id = self.create_serie_id(table_name, base_name, device_name,
                                        soc, target_abi, runtime)
        if is_test:
            succ = self.report_test(serie_id, value)
        else:
            succ = self.report_benchmark(serie_id, value, trend)
        return succ

    def get_and_create_build_from_git(self, project_id=MACE_PROJECT_ID):
        commit_id = sh.git('log', '--no-merges', '-n1',
                           '--pretty=format:%H', _tty_out=False).strip()
        abbrev_commit_id = sh.git('rev-parse', "%s~1" % commit_id,
                                  _tty_out=False).strip()
        author_name = sh.git('log', '--pretty=format:%an',
                             '-1', commit_id, _tty_out=False).strip()
        author_mail = sh.git('log', '--pretty=format:%ae',
                             '-1', commit_id, _tty_out=False).strip()
        subject = sh.git('log', '--pretty=format:%s',
                         '-1', commit_id, _tty_out=False).strip()
        return self.get_and_create_build(commit_id, abbrev_commit_id,
                                         author_name, author_mail,
                                         subject, project_id)

    def get_and_create_build(self, commit_id, abbrev_commit_id, author_name,
                             author_mail, subject, project_id=MACE_PROJECT_ID):
        if self._build_id is None:
            build_id = self.create_build_id()
            succ = self._dana_cli.add_build(project_id=project_id,
                                            build_id=build_id,
                                            build_hash=commit_id,
                                            abbrev_hash=abbrev_commit_id,
                                            author_name=author_name,
                                            author_mail=author_mail,
                                            subject=subject)
            if succ:
                self._build_id = build_id
        return self._build_id

    def create_build_id(self):
        if False:
            latest_id = sh.git('log', '-n1', '--pretty=format:%H',
                               _tty_out=False).strip()
            timestamp = sh.git('log', '--pretty=format:%ad', '-1',
                               "--date=format:%Y-%m-%d %H:%M:%S",
                               latest_id, _tty_out=False).strip()
            unix_time = time.mktime(
                time.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
            build_id = int(unix_time)
        else:
            build_id = int(time.time())
        return build_id

    def create_serie_id(self, table_name, base_name, device_name,
                        soc, target_abi, runtime):
        base_serie_id = "%s_%s_%s_%s_%s" % (base_name, device_name,
                                            soc, target_abi, runtime)
        return self.create_serie_id_lite(table_name, base_serie_id)

    def create_serie_id_lite(self, table_name, base_name):
        serie_id = "%s_%s" % (table_name, base_name)
        serie_id = re.sub(r'\s+', '', serie_id.strip())
        return serie_id

    def report_benchmark(self, serie_id, value, trend,
                         project_id=MACE_PROJECT_ID):
        if not self._dana_cli.serie_available(serie_id):
            succ = self._dana_cli.add_benchmark_serie(
                project_id=project_id, serie_id=serie_id,
                range=DEFAULT_RANGE, required=DEFAULT_REQUIRED,
                trend=trend)
            if not succ:
                print("Add benchmark serie_id(%s) failed." % serie_id)
                return False
        build_id = self.get_and_create_build_from_git()
        if build_id is None:
            print("Add build id failed.")
            return False
        succ = self._dana_cli.add_sample(project_id=project_id,
                                         serie_id=serie_id,
                                         build_id=build_id,
                                         value=value)
        return succ

    def report_test(self, serie_id, value,
                    project_id=MACE_PROJECT_ID):
        if not self._dana_cli.serie_available(serie_id):
            succ = self._dana_cli.add_test_serie(project_id=project_id,
                                                 serie_id=serie_id)
            if not succ:
                print("Add test serie_id(%s) failed." % serie_id)
                return False
        build_id = self.get_and_create_build_from_git()
        if build_id is None:
            print("Add build id failed.")
            return False
        succ = self._dana_cli.add_sample(project_id=project_id,
                                         serie_id=serie_id,
                                         build_id=build_id,
                                         value=value)
        return succ

    def service_available(self):
        cur_time = int(time.time())
        if self._valid_time is None or \
                cur_time - self._valid_time > SERVICE_LIFE:
            self._service_available = self._dana_cli.service_available()
            self._valid_time = cur_time
        return self._service_available


if __name__ == '__main__':
    DanaUtil().fast_report_benchmark(
        base_name="%s-%s" % ('init', "model_name"),
        device_name='device_name',
        soc='target_socs',
        target_abi='target_abi',
        runtime='device_type', value=90)
