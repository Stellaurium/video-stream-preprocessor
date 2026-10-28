"""Exercise actual PowerShell setup selection without creating an environment."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == 'nt', 'Windows setup entry')
class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'setup path with spaces'
        (self.root / 'tools').mkdir(parents=True)
        self.script = self.root / 'tools/setup.ps1'
        shutil.copy2(Path(__file__).resolve().parents[1] / 'tools/setup.ps1', self.script)

    def run_setup(self, *args, env=None):
        return subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass',
                               '-File',str(self.script),'-CheckOnly','-NoPause',*args],
                              capture_output=True, timeout=15, env=env)

    def test_explicit_python_checked_without_installing_or_creating_venv(self):
        result = self.run_setup('-Python', sys.executable)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        report = json.loads(result.stdout.decode('utf-8-sig'))
        self.assertEqual(report['version'], '3.11')
        self.assertEqual(report['bits'], 64)
        self.assertFalse((self.root / '.venv').exists())

    def test_wrong_version_is_rejected_without_falling_back(self):
        wrong = self.root / 'fake-python.cmd'
        wrong.write_bytes(b'@echo off\r\necho 2^|7^|64^|C:\\fake\\python.exe\r\n')
        result = self.run_setup('-Python', str(wrong))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'Python 3.11 x64 is required', result.stderr)
        self.assertFalse((self.root / '.venv').exists())

    def test_missing_explicit_executable_reports_failure(self):
        result = self.run_setup('-Python', str(self.root / 'not-installed.exe'))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / '.venv').exists())

    def test_app_python_override_works_from_another_working_directory(self):
        result = self.run_setup(env={**os.environ, 'APP_PYTHON':sys.executable})
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertEqual(json.loads(result.stdout.decode('utf-8-sig'))['bits'], 64)
