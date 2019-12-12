FROM python:3.7
ENV USER=dockeruser
ENV HOME="/home/$USER"
RUN pip install gym matplotlib pybullet torch tensorflow==1.14 numpy numba scipy pandas quaternion networkx opencv-python pyyaml imageio tqdm tables scikit-image
RUN git clone https://github.com/openai/baselines.git
RUN pip install -e baselines/
RUN useradd -m -d $HOME $USER
RUN chown -R $USER $HOME
USER $USER
WORKDIR $HOME
CMD ["/bin/bash"]
