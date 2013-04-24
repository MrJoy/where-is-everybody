#!/bin/bash
if [[ $THOROUGH -gt 0 ]]; then
  export ATTEMPTS=1000
elif [[ $ATTEMPTS -eq 0 ]]; then
  export ATTEMPTS=1
fi

export FAIL=0
echo "TEST: Hammering 'bin/randoms --first --last --samples=67108864 --blocks=1 --seed=1' $ATTEMPTS times to try and reproduce device-allocation error."
for i in {1..$ATTEMPTS}; do
  if [[ ! -x bin/randoms ]]; then
    echo "FAIL: Need to run 'make all' first."
    exit 1
  fi
  RESULT=$(bin/randoms --first --last --samples=67108864 --blocks=1 --seed=1 2>&1 |
    fgrep -v 'Block #' |
    fgrep -v 'Generating 1 block(s) of 67108864 samples.' |
    fgrep -v 'Using seed: 1' |
    grep -E '^ERROR:')

  if [[ $RESULT != '' ]]; then
    echo $RESULT
    echo "FAIL: Got an unexpected error for a case that should pretty much always work."
    exit 1
  fi
done

echo "PASS: Got no errors in bin/randoms after $ATTEMPTS attempt(s)."
