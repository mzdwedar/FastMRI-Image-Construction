import os,sys


def test_ci_drill_fails():
    assert 1 == 2, "intentional failure to verify CI catches it"
