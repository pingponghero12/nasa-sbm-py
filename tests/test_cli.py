"""CLI tests."""

import pytest
import subprocess
import os
from pathlib import Path


def test_cli_help():
    result = subprocess.run(['nasa-sbm', '--help'], capture_output=True, text=True)

    assert result.returncode == 0
    assert 'NASA Standard Breakup Model' in result.stdout


def test_cli_explosion_help():
    result = subprocess.run(['nasa-sbm', 'explosion', '--help'], capture_output=True, text=True)

    assert result.returncode == 0
    assert '--mass' in result.stdout



def test_cli_collision_help():
    result = subprocess.run(['nasa-sbm', 'collision', '--help'], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert '--mass1' in result.stdout
    assert '--velocity' in result.stdout


def test_cli_explosion_run(tmp_path):
    output_file = tmp_path / "test_explosion.nc"
    
    result = subprocess.run([
        'nasa-sbm', 'explosion',
        '--mass', '100',
        '--cutoff', '0.1',
        '--seed', '42',
        '--out', str(output_file)
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert output_file.exists()
    assert 'Generated' in result.stdout


def test_cli_collision_run(tmp_path):
    output_file = tmp_path / "test_collision.nc"
    
    result = subprocess.run([
        'nasa-sbm', 'collision',
        '--mass1', '100',
        '--mass2', '200',
        '--velocity', '10.0',
        '--cutoff', '0.1',
        '--seed', '42',
        '--out', str(output_file)
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert output_file.exists()
    assert 'Generated' in result.stdout


def test_cli_explosion_with_options(tmp_path):
    output_file = tmp_path / "test_rocket. nc"
    
    result = subprocess.run([
        'nasa-sbm', 'explosion',
        '--mass', '839',
        '--sat-type', 'rocket_body',
        '--cutoff', '0.05',
        '--out', str(output_file),
        '--enforce-mass-conservation'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert output_file.exists()
