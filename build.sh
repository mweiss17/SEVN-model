docker run \
-v /home/martin/code/SEVN-model:/home/dockeruser/SEVN-model/ \
-v /home/martin/code/SEVN:/home/dockeruser/SEVN/ \
-v /home/martin/data/sevn-data:/home/dockeruser/data/sevn-data/ \
-v /home/martin/data/trained_models/:/home/dockeruser/data/trained_models/ \
sevn \
bash /home/dockeruser/SEVN-model/run.sh