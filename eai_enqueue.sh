MY_ACC="9501fb28-d2f5-4dff-88de-ecf0d2ed10a0"
TAG="registry.console.elementai.com/$MY_ACC/test"
eai data upload --id mila.martin.sevnmodel .

eai job submit \
--gpu 1 \
-i $TAG \
--preemptable --restartable \
--data mila.martin.sevnmodel:/home/dockeruser/SEVN-model/ \
--data mila.martin.sevn:/home/dockeruser/SEVN/ \
--data mila.martin.sevndata:/home/dockeruser/data/sevn-data/ \
--data mila.martin.trainedmodels:/home/dockeruser/data/trained_models/ \
bash /home/dockeruser/SEVN-model/run.sh