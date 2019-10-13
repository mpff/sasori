# To create this image you need to first install docker (www.docker.com)
# Then, from your command line and being in this folder, run

# docker build -t animerec_ide .

# After the docker image has been built you can run it by

# docker run -d -p 8787:8787 -e PASSWORD=??? --name sposm sposm_ide

# Do not forget to set a passord, as depending on your configuration
# the container might be accessible in the network!

# Access your new integrated development environment in your browser

# http://localhost:8787

# When you are done, you can stop the container instance

# docker stop animerec

# And (maybe) delete it (ATTENTION: Everything you did in the container is lost)

# docker rm animerec

# If you also want to delete the image that was created from the Dockerfile

# docker image rm animerec_ide

# Below you will find the actual Dockerfile

FROM ubuntu:18.04

MAINTAINER Manuel Pfeuffer "pfeuferm@hu-berlin.de"

RUN apt-get update 
RUN apt-get install python3-pip
RUN pip3 install — upgrade pip

# The below invalidates the cache whenever the Head of the Github repo changes
ADD https://api.github.com/repos/mpff/animerec/git/refs/heads/master version.json

RUN sudo git clone https://github.com/mpff/animerec.git

RUN pip3 install -r ~/animerec/requirements.txt

WORKDIR ~/animerec
