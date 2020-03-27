#!/usr/bin/env bash

. ./setup.sh

commands=(run install update test debug bandit version pypi testpypi)
usage="Usage: $0 <command>
  commands: ${commands[*]}"


version_update () {
  version=$(grep "^version" pyproject.toml | cut -d '"' -f2)
  echo "__version__ = '$version'" > ./src/phys_learning/__version__.py
  sed -i.bak "s/    assert __version__ == '.*'/    assert __version__ == '$version'/" tests/test_phys_learning.py
  rm -f tests/test_phys_learning.py.bak
  git tag -a v"${version}"
}

case $1 in
  run)poetry run phys_learning run;;
  install)poetry install;;
  update)poetry update;;
  test)poetry run pytest -v ./tests/;;
  bandit)mkdir -p report/;poetry run bandit -r ./src/phys_learning/ -f html > report/bandit.html;;
  version)version_update;;
  pypi)version_update; poetry publish --build;;
  testpypi)version_update; poetry publish -r testpypi --build;;
  "")echo "$usage";exit 0;;
  *)echo "$usage";exit 1;;
esac
