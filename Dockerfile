FROM python:3.7

ENV USER=dockeruser
ENV HOME="/home/$USER"
RUN useradd -m -d $HOME $USER
RUN chown -R $USER $HOME
USER $USER
RUN pip install --user gym  matplotlib pybullet torch tensorflow==1.14 numpy numba scipy pandas quaternion networkx opencv-python pyyaml imageio tqdm tables
WORKDIR $HOME
RUN git clone https://github.com/openai/baselines.git
RUN pip install --user -e baselines/
RUN pip install --user scikit-image
CMD ["/bin/bash"]
