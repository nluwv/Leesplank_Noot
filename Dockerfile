FROM registry.suse.com/bci/bci-base:15.5.36.14.13

#
# Disable BCI repros
#

RUN set -eux ; \
  sed -i 's#enabled=1#enabled=0#g' /etc/zypp/repos.d/SLE_BCI.repo

RUN set -eux ; \
  zypper -n addrepo http://download.opensuse.org/distribution/leap/15.5/repo/oss/ myrepo3 ; \
  echo 'gpgcheck=0' >> /etc/zypp/repos.d/myrepo3.repo ; \
  zypper -n addrepo https://download.opensuse.org/repositories/devel:/languages:/perl/SLE_15_SP5 myrepo4 ; \
  echo 'gpgcheck=0' >> /etc/zypp/repos.d/myrepo4.repo

RUN set -eux ; \
  sed -i 's#gpgcheck=1#gpgcheck=0#g' /etc/zypp/repos.d/*.repo

#
# Install build dependencies
#
RUN set -eux; \
  zypper -n refresh ; \
  zypper --no-gpg-checks -n install -y --force-resolution \
    git cmake gcc12 gcc12-c++ gcc12-fortran zlib-devel numactl awk patch tar autoconf automake libtool libjson-c-devel graphviz ncurses-devel nano which libjansson4 libnl3-200; \
  zypper clean

#ENV ROCM_RPM https://repo.radeon.com/amdgpu-install/6.2.3/sle/15.5/amdgpu-install-6.2.60203-1.noarch.rpm
#ENV ROCM_RELEASE 6.2.3
ENV ROCM_RPM https://repo.radeon.com/amdgpu-install/6.3.1/sle/15.5/amdgpu-install-6.3.60301-1.noarch.rpm
ENV ROCM_RELEASE 6.3.1

# Add the devel repo to install libtbb12 required by rocrand
RUN set -eux ; \
  zypper -n addrepo https://download.opensuse.org/repositories/devel:/libraries:/c_c++/15.5/ myrepo5 ; \
  echo 'gpgcheck=0' >> /etc/zypp/repos.d/myrepo5.repo

RUN set -eux ; \
  zypper --no-gpg-checks -n install $ROCM_RPM

RUN set -eux ; \
  sed -i 's#gpgcheck=1#gpgcheck=0#g' /etc/zypp/repos.d/*.repo

RUN set -eux ; \
  zypper --no-gpg-checks -n install --oldpackage libsystemd0-249.16 libudev1-249.16

RUN set -eux ; \
  zypper refresh ; \
  amdgpu-install -y --no-dkms --usecase=rocm --rocmrelease=$ROCM_RELEASE ; \
  zypper cc -a

RUN set -eux ; \
  zypper --no-gpg-checks -n install -y --force miopen-hip-gfx942kdb

#
# ROCm environment
#
ENV ROCM_PATH /opt/rocm-$ROCM_RELEASE
ENV PATH $ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROCM_PATH/lib

#
# Install miniconda
#
RUN set -eux ; \
  curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py312_24.5.0-0-Linux-x86_64.sh ; \
  bash ./Miniconda3-* -b -p /opt/miniconda3 -s ; \
  rm -rf ./Miniconda3-*

ENV WITH_CONDA "source /opt/miniconda3/bin/activate base"


#
# AWS plug in
#
ENV LIBFABRIC_PATH /opt/cray
ENV MPICH_PATH "/opt/cray/pe/mpich/8.1.29/ofi/crayclang/16.0"
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$LIBFABRIC_PATH/lib64:$MPICH_PATH/lib:/opt/cray/pe/lib64:/opt/cray/pe/lib64/cce
ENV REMOVE_CRAY_DEPS 'rm -rf /opt/cray /usr/lib64/libcxi*'

RUN --mount=type=bind,source=cray.tar,dst=/cray.tar \
  set -eux ; \
  cd / ; \
  tar -xf cray.tar ; \
  \
  git clone -b cxi https://github.com/rocm/aws-ofi-rccl /opt/mybuild ; \
  cd /opt/mybuild ; \
  git checkout -b mydev 17d41cb ; \
  ./autogen.sh ; \
  \
  cd /opt/mybuild ; \
  export CPATH=$LIBFABRIC_PATH/include ; \
  export LIBRARY_PATH=$LD_LIBRARY_PATH ; \
  LDFLAGS='-lcxi' CC=gcc-12 ./configure --with-libfabric=$LIBFABRIC_PATH --enable-trace --with-hip=$ROCM_PATH --with-rccl=$ROCM_PATH/rccl --disable-tests ; \
  LDFLAGS='-lcxi' CC=gcc-12 nice make -j ; \
  \
  mkdir /opt/aws-ofi-rccl ; \
  mv src/.libs/librccl-net.so* /opt/aws-ofi-rccl ; \
  rm -rf /opt/mybuild ; \
  ldd /opt/aws-ofi-rccl/librccl-net.so ; \
  $REMOVE_CRAY_DEPS

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/aws-ofi-rccl

#
# RCCL tests
#
RUN --mount=type=bind,source=cray.tar,dst=/cray.tar \
  set -eux ; \
  cd / ; \
  tar -xf cray.tar ; \
  \
  git clone https://github.com/rocm/rccl-tests /opt/mybuild ; \
  sed -i 's/-std=c++14/-std=c++14 --amdgpu-target=gfx90a:xnack- --amdgpu-target=gfx90a:xnack+/g' /opt/mybuild/src/Makefile ; \
  \
  cd /opt/mybuild ; \
  CC=gcc-12 \
    CXX=g++-12 \
    MPI_HOME=$MPICH_PATH \
    ROCM_PATH=$ROCM_PATH \
    MPI=1 \
    NCCL_HOME=$ROCM_PATH/rccl \
    nice make -j ; \
  mkdir /opt/rccltests ; \
  mv /opt/mybuild/build/* /opt/rccltests ; \
  rm -rf /opt/mybuild ; \
  $REMOVE_CRAY_DEPS

#
# Install conda environment
#
ENV PYTHON_VERSION 3.12
RUN $WITH_CONDA; set -eux ; \
  conda create -n pytorch python=$PYTHON_VERSION ; \
  conda activate pytorch ; \
  conda install -y ninja pillow cmake pyyaml
ENV WITH_CONDA "source /opt/miniconda3/bin/activate pytorch"

ENV PYTORCH_ROCM_ARCH gfx942
ENV PYTORCH_VERSION '2.4.0+rocm6.3.1'

RUN $WITH_CONDA; set -eux ; \
  pip3 install --pre torch==${PYTORCH_VERSION} -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1/ ; \
  cp /opt/rocm/lib/librccl.so $(dirname $(python -c 'import torch; print(torch.__file__)'))/lib

ENV APEX_VERSION 1.4.0+rocm6.3.1
RUN $WITH_CONDA; set -eux ; \
  pip3 install --pre apex==${APEX_VERSION} -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1


ENV TORCHVISION_VERSION 0.19.0+rocm6.3.1
RUN $WITH_CONDA; set -eux ; \
  pip3 install --pre torchvision==$TORCHVISION_VERSION -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1

ENV TORCHAUDIO_VERSION 2.4.0+rocm6.3.1
RUN $WITH_CONDA; set -eux ; \
  pip3 install --pre torchaudio==${TORCHAUDIO_VERSION} -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3.1

#
# AMD-SMI
#
RUN $WITH_CONDA; set -eux ; \
  cd $ROCM_PATH/share/amd_smi ; \
  python3 -m pip wheel . --wheel-dir=/opt/wheels ; \
  pip install /opt/wheels/amdsmi-*.whl

#
# Bits and Bytes and its dependencies
#

RUN $WITH_CONDA; set -eux ; \
  cd / ; \
  rm -rf /opt/bitsandbytes ; \
  mkdir /opt/bitsandbytes ; \
  git clone -b rocm_enabled_multi_backend --recursive https://github.com/ROCm/bitsandbytes /opt/bitsandbytes ; \
  cd /opt/bitsandbytes ; \
  pip install -r requirements-dev.txt ; \
  cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -DAMDGPU_TARGETS="gfx942" -S . ; \
  nice make -j ; \
  python3 -m pip wheel --no-deps . --wheel-dir=/opt/wheels ; \
  pip install /opt/wheels/bitsandbytes-*.whl


RUN $WITH_CONDA; set -eux ; \
   pip install transformers sentencepiece python-dotenv accelerate google-api-python-client

ENTRYPOINT ["/opt/miniconda3/bin/conda", "run", "--no-capture-output", "-n", "pytorch"]

#podman run -it --device=/dev/dri --device=/dev/kfd  --network=host --ipc=host --group-add keep-groups -v $HOME:/workdir -v /opt/cray:/opt/cray -v /usr/lib64/libcxi.so.1.5.0:/usr/lib64/libcxi.so.1.5.0 --workdir /workdir sam631 bash