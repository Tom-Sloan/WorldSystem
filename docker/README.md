nc -vz 127.0.0.1 9876
ssh -NT -R 127.0.0.1:9876:localhost:9876 sam3@134.117.167.139
RERUN_BIND=0.0.0.0:9876 rerun viewer