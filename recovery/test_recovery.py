"""CPU contracts for configuration, default-off loading and owned cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(path):
    spec = importlib.util.spec_from_file_location("recovery_test_target", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class RecoveryTests(unittest.TestCase):
    def test_unpinned_provider_fails_before_loading_framework(self):
        provider = module(ROOT / "vllm_omni/diffusion/layers/lingbot_recovery.py")
        with patch.dict(os.environ, {"LINGBOT_RECOVERY_MANIFEST":"", "LINGBOT_RECOVERY_MANIFEST_SHA256":""}):
            with self.assertRaisesRegex(RuntimeError, "explicit manifest"):
                provider.load()
        self.assertIsNone(provider._BOUND)

    def test_config_has_no_lost_paths_and_mutually_exclusive_providers(self):
        import yaml
        with tempfile.TemporaryDirectory() as folder:
            for candidate in (False,True):
                path=Path(folder)/"config.yaml"
                subprocess.run([sys.executable,str(ROOT/"recovery/make_config.py"),"--output",str(path)]
                               + (["--candidate"] if candidate else []),check=True)
                self.assertNotIn("/mnt/data",path.read_text())
                config=yaml.safe_load(path.read_text())["stages"][0]["model_config"]
                self.assertEqual(config["lingbot_npu_recovery_qk"],candidate)
                self.assertEqual(config["lingbot_npu_fixed_timestep_bank"],candidate)
                self.assertFalse(config["lingbot_npu_qk_single_host_admission_v1"])
                self.assertFalse(config["lingbot_npu_vae_exact_pack"])

    def test_fatal_reclaims_only_the_owned_eight_workers(self):
        supervisor=module(ROOT/"recovery/supervise.py")
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            service=folder/"service.py"
            service.write_text('''import json,os,subprocess,sys,time
from pathlib import Path
workers=[subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]) for _ in range(8)]
run=Path(os.environ["ROOT_LINGBOT_SP8_FATAL_DIR"])
(run/"workers.json").write_text(json.dumps([p.pid for p in workers]))
time.sleep(1)
fields=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
record=dict(nonce=os.environ["ROOT_LINGBOT_SP8_FATAL_NONCE"],pid=os.getpid(),
 process_start_ticks=int(fields[19]),restart_scope="all8",
 boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
(run/"fatal-test.json").write_text(json.dumps(record))
time.sleep(60)
''')
            outsider=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True)
            try:
                result=subprocess.run([sys.executable,str(ROOT/"recovery/supervise.py"),"--runs",str(folder/"runs"),
                                       "--",sys.executable,str(service)],capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,1,result.stderr)
                run=Path(result.stdout.strip())
                receipt=json.loads((run/"result.json").read_text())
                self.assertTrue(receipt["fatal"])
                self.assertEqual(receipt["remaining_owned_processes"],[])
                for pid in json.loads((run/"workers.json").read_text()):
                    self.assertIsNone(supervisor.identity(pid))
                self.assertIsNone(outsider.poll())
            finally:
                outsider.terminate(); outsider.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
