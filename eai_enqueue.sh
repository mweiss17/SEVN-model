MY_ACC="9501fb28-d2f5-4dff-88de-ecf0d2ed10a0"
TAG="registry.console.elementai.com/$MY_ACC/test"

# build container
docker build -f Dockerfile -t sevn .

# login to EAI's dockerhub
$(eai docker get-login)

# push it to EAI registry
docker push $TAG

# upload EAI datas
eai data upload --id mila.martin.sevnmodel .
#eai data upload --id mila.martin.sevn ../SEVN
#eai data upload --id mila.martin.sevndata ../../data/sevn-data
#eai data upload --id mila.martin.trainedmodels ../../data/trained_models

# Run batch job
#eai job submit \
#--gpu 1 \
#-i $TAG \
#--preemptable --restartable \
#--data mila.martin.sevnmodel:/home/dockeruser/SEVN-model/ \
#--data mila.martin.sevn:/home/dockeruser/SEVN/ \
#--data mila.martin.sevndata:/home/dockeruser/data/sevn-data/ \
#--data mila.martin.trainedmodels:/home/dockeruser/data/trained_models/ \
#bash /home/dockeruser/SEVN-model/run.sh

# Run interactive job
eai job submit \
--gpu 1 \
-I \
-i $TAG \
--data mila.martin.sevnmodel:/home/dockeruser/SEVN-model/ \
--data mila.martin.sevn:/home/dockeruser/SEVN/ \
--data mila.martin.sevndata:/home/dockeruser/data/sevn-data/ \
--data mila.martin.trainedmodels:/home/dockeruser/data/trained_models/ \
sleep 3600
eai job exec --last bash